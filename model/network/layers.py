import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from timm.models.vision_transformer import Block
from functools import partial

class AttentionPool(nn.Module):
    def __init__(self, dim, num_heads=24, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, repeats=1, mask=None):

        B, N, C = x.shape
        q = self.q(x[:,:repeats]).reshape(B, repeats, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale
        
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) 
        if mask is not None:
            eye = torch.eye(repeats,device=x.device).unsqueeze(0).repeat(B,1,1).bool()
            mask_temp = torch.cat([eye, mask],dim=-1).unsqueeze(1)
            attn = attn.masked_fill_(~mask_temp.bool(), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_cls = (attn @ v).transpose(1, 2).reshape(B, repeats, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

class AttentionPoolBlock(Block):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        
        dim = kwargs["dim"]
        num_heads = kwargs["num_heads"]
        self.attn = AttentionPool(dim,num_heads)

    def forward(self, cls_tokens, x, repeats=1,mask=None):
        x = torch.cat((cls_tokens,x),dim=1)
        att = self.attn(self.norm1(x),repeats=repeats,mask=mask)
        cls_tokens = cls_tokens + self.drop_path1(self.ls1(att))
        cls_tokens = cls_tokens + self.drop_path2(self.ls2(self.mlp(self.norm2(cls_tokens))))
        return cls_tokens
        
class AttPool(nn.Module):
    def __init__(self, in_dim, num_layers=2):
        super().__init__()
        num_heads = int(in_dim // 32) ###
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.blocks = nn.ModuleList([
            AttentionPoolBlock(dim=in_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for i in range(num_layers)])
        self.norm = partial(nn.LayerNorm, eps=1e-6)(in_dim)
        self.apply(self._init_weights)
        self.embed_dim = in_dim
        
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, in_dim, 6, 6))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, repeats=1, **kwargs):
        cls_tokens = self.cls_token.repeat(x.shape[0], repeats, 1)
    
        w = int(x.size(1) ** 0.5)
        absolute_pos_embed = torch.nn.functional.interpolate(self.absolute_pos_embed, size=(w, w), mode='bicubic')
        absolute_pos_embed = absolute_pos_embed.flatten(2).transpose(1, 2) 
        x = x + absolute_pos_embed
        for i, blk in enumerate(self.blocks):
            cls_tokens = blk(cls_tokens,x,repeats,**kwargs)
        cls_tokens = self.norm(cls_tokens)        
        return cls_tokens.squeeze(1)
    
class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
        use_attpool=False,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if use_attpool:
            self.att = AttPool(in_dim)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pool=False,**kwargs):
        x = self.forward_first(x,pool,**kwargs)
        return self.forward_last(x)
    
    def forward_first(self,x,pool=False,**kwargs):
        if pool:
            x = self.att(x,**kwargs)
        return self.mlp(x)
    
    def forward_last(self, x):
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x

def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)