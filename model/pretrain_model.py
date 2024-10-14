import torch
import torch.nn as nn
import numpy as np
import kornia.augmentation as korniatfm
from .network.swin_pretrain import SWIN
from .network.layers import DINOHead
from .network.loss import DINOLoss, iBOTPatchLoss
import math
from omegaconf import OmegaConf
from functools import partial
import random
from torch_scatter import scatter_mean
from copy import deepcopy

class Transform(nn.Module):
    def __init__(self, cfg, downsample=32):
        
        super().__init__()
       
        global_crops_scale = cfg.get("global_crops_scale", [0.32,1.0])
        local_crops_scale = cfg.get("local_crops_scale", [0.05,0.32])
        global_crops_size = cfg.get("global_crops_size", 192)
        local_crops_size = cfg.get("local_crops_size", 96) 
        input_size = cfg.get("input_size", 224)
        same_on_batch = False
        
        self.input = korniatfm.RandomCrop((input_size,input_size))
        
        teacher1 = [korniatfm.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice",align_corners=True)] 
        teacher1 += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
        
        teacher2 = [korniatfm.RandomResizedCrop((global_crops_size, global_crops_size), scale=global_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice",align_corners=True)] 
        teacher2 += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
        
        
        student = [korniatfm.RandomResizedCrop((local_crops_size, local_crops_size), scale=local_crops_scale, same_on_batch=same_on_batch,cropping_mode="slice", align_corners=True)] 
        student += [korniatfm.RandomHorizontalFlip(p=0.5,same_on_batch=same_on_batch)]
    
        self.student = student
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.local_crops_size = local_crops_size
        self.global_crops_size = global_crops_size
        self.input_size = input_size
        grid_h, grid_w = np.meshgrid(range(input_size),range(input_size), indexing='ij')
        grid = torch.from_numpy(np.stack([grid_w, grid_h])).float().unsqueeze(0)
        self.register_buffer("grid",grid + 1)
        self.downsample = downsample
        
    @torch.no_grad()
    def forward(self, img):
        
        img = self.input(img)
        
        assert self.input_size == img.size(2) and self.input_size == img.size(3), f"{img.size()}"
        
        b = img.size(0)
        teacher = img.repeat(2,1,1,1)
        t_grid = self.grid.expand(b * 2,-1,-1,-1)
        
        for f in self.teacher1:
            teacher = f(teacher)
            t_grid = f(t_grid,params=f._params)
          
        t_grid = t_grid - 1
        t_grid = 2.0 * t_grid/ max(self.input_size - 1, 1) - 1.0
        t_grid = torch.nn.functional.interpolate(t_grid, self.global_crops_size//self.downsample,mode="bilinear", align_corners=True) #torch.nn.functional.adaptive_avg_pool2d(t1_grid, self.global_crops_size//32)
        t_grid = t_grid.permute(0, 2, 3, 1)
        
        
        student = img.repeat(8,1,1,1)
        s_grid = self.grid.expand(b * 8,-1,-1,-1)
        
        student, s_grid = self.forward_imp(student, s_grid)
        output = dict(
            img = img,
            global_crops = teacher,
            local_crops = student,
            s_grid = s_grid,
            t_grid = t_grid,
            )
        return collate_data(output)
    
    def forward_imp(self, s, g):
        
        downsample = partial(torch.nn.functional.interpolate, size = self.local_crops_size//self.downsample, mode="bilinear", align_corners=True) #partial(torch.nn.functional.adaptive_avg_pool2d, output_size = self.local_crops_size//32)
        
        for f in self.student:
            s = f(s)
            g = f(g,params=f._params)
            
        g = g - 1
        g = 2.0 * g/ max(self.input_size - 1, 1) - 1.0
        g = downsample(g) 
        g = g.permute(0, 2, 3, 1)

        return s, g

@torch.no_grad()
def batch_cosine_KMeans(X,num_clusters=6,max_iter=10):
    X = X.clone().detach()
    X = torch.nn.functional.normalize(X, dim=2)
    N, L, D = X.shape 
    noise = torch.rand(N, L, device=X.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :num_clusters]
    centroids = torch.gather(X, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    for _ in range(max_iter):
        centroids = torch.nn.functional.normalize(centroids, dim=2)
        dis = 1 - torch.einsum('bij,bkj->bik', X, centroids)
        assignments = torch.argmin(dis, dim=2)
        centroids = scatter_mean(X, assignments,dim=1)
    return assignments, centroids


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        return mask

class BaseModel(nn.Module):
    def __init__(self, cfg=None, n_tokens=8, *args,**kwargs):
        super().__init__()
        global collate_data
        
        cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        self.n_tokens = n_tokens
        student_model_dict = dict()
        teacher_model_dict = dict()

        t_kwargs = deepcopy(kwargs)
        t_kwargs["drop_path_rate"] = 0.0
        student_backbone, teacher_backbone = SWIN(*args,**kwargs), SWIN(*args,**t_kwargs)
        embed_dim = kwargs['embed_dim']  * 2**3

            
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        print(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        print("OPTIONS -- DINO")
        
        print(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
        print(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
        print(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
        print(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
        self.dino_loss_weight = cfg.dino.loss_weight
        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
            use_attpool = True,
        )
        self.dino_loss = DINOLoss(self.dino_out_dim)

        student_model_dict["dino_head"] = dino_head()
        teacher_model_dict["dino_head"] = dino_head()

        print("OPTIONS -- IBOT")
        print(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        print(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        print(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")

        self.ibot_loss_weight = cfg.ibot.loss_weight
        assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
        assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
        self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
        self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
        if self.ibot_separate_head:
            print(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
            print(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
            print(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
            print(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
            ibot_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=cfg.ibot.head_hidden_dim,
                bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                nlayers=cfg.ibot.head_nlayers,
            )
            student_model_dict["ibot_head"] = ibot_head()
            teacher_model_dict["ibot_head"] = ibot_head()
        else:
            print("OPTIONS -- IBOT -- head shared with DINO")

        student_model_dict["cluster_head"] = dino_head()
        teacher_model_dict["cluster_head"] = dino_head()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.prepare_for_distributed_training()
        
        def collate_data(samples, mask_ratio_tuple = cfg.ibot.mask_ratio_min_max, mask_probability=cfg.ibot.mask_sample_probability): 
    
            collated_global_crops = samples["global_crops"]
            collated_local_crops = samples["local_crops"]
            img = samples["img"]
            s_grid = samples["s_grid"]
            t_grid = samples["t_grid"]
            device = collated_global_crops.device
        
            B = len(collated_global_crops)
            N = self.n_tokens**2  
            n_samples_masked = int(B * mask_probability)
            probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
            upperbound = 0
            masks_list = []
            for i in range(0, n_samples_masked):
                prob_min = probs[i]
                prob_max = probs[i + 1]
                masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
                upperbound += int(N * prob_max)
            for i in range(n_samples_masked, B):
                masks_list.append(torch.BoolTensor(mask_generator(0)))
        
            random.shuffle(masks_list)
        
            collated_masks = torch.stack(masks_list).flatten(1).to(device)
            mask_indices_list = collated_masks.flatten().nonzero().flatten()
        
            masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
        
            return {
                "collated_global_crops": collated_global_crops,
                "collated_local_crops": collated_local_crops,
                "s_grid": s_grid,
                "t_grid": t_grid,
                "img": img,
                "collated_masks": collated_masks,
                "mask_indices_list": mask_indices_list,
                "masks_weight": masks_weight,
                "upperbound": upperbound,
                "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long).to(device),
            }
                

class PretrainModel(BaseModel):
    def __init__(self, num_clusters = 8, aug_downsample=16, *args, **kwargs):
        global mask_generator
        N = kwargs["cfg"]["crops"]["global_crops_size"] // aug_downsample
        mask_dim = N
        mask_generator = MaskingGenerator(
            input_size=(N, N),
            max_num_patches=0.5 * N * N,
        )
        super().__init__(mask_dim=mask_dim,*args,**kwargs)
        self.num_clusters = num_clusters
        self.aug = Transform(kwargs["cfg"]["crops"], aug_downsample)

    def forward(self, img, m, teacher_temp):
        self.update_teacher(m)
        images = self.aug(img)
        return self.forward_imp(images,teacher_temp)
        
    def forward_imp(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"]
        local_crops = images["collated_local_crops"]
        t_grid = images["t_grid"]
        imgs = images["img"]
        
        masks = images["collated_masks"]
        mask_indices_list = images["mask_indices_list"]
        n_masked_patches_tensor = images["n_masked_patches"]
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"]

        n_local_crops_loss_terms =  n_local_crops * n_global_crops 
        n_global_crops_loss_terms = n_global_crops + n_global_crops 

        do_dino = self.do_dino
        do_ibot = self.do_ibot
    
        ibot_loss_scale = 1.0 / n_global_crops

        b = global_crops.size(0)
        
        def tomask(x,maps):
            return x.scatter_(1, maps.unsqueeze(1).long(), 1).bool()
        
        KEY = "x_norm_patchtokens" 
        
        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            global t_assign_map, t_mask, s_assign_map, s_mask, teacher_patchtokens 
            
            teacher_cls_tokens = self.teacher.backbone(imgs, is_training=True)[KEY]
            wt = int(teacher_cls_tokens.size(1)**0.5)
            
            assign, _ = batch_cosine_KMeans(teacher_cls_tokens, num_clusters=self.num_clusters)
            attmask = tomask(torch.zeros((b//2,self.num_clusters, wt**2),  device=teacher_cls_tokens.device),assign)
            teacher_cls_tokens = self.teacher.cluster_head.att(teacher_cls_tokens,self.num_clusters,mask=attmask)
            
            t_assign = assign.repeat(n_global_crops,1)
            t_assign_map = torch.zeros((b,self.num_clusters, wt, wt),  device=teacher_cls_tokens.device).scatter_(1, t_assign.view(b,wt,wt).long().unsqueeze(1), 1)
            t_assign_map = nn.functional.grid_sample(t_assign_map, t_grid, align_corners=True)
            t_assign_map = (t_assign_map > .25).to(t_assign_map) 
            t_mask = t_assign_map.view(b,self.num_clusters,-1).sum(-1) >= .999
            
            
            teacher_cls_tokens = teacher_cls_tokens.flatten(0,1)
            n_clusters = teacher_cls_tokens.size(0)
            x = self.teacher.backbone(global_crops, is_training=True)
            assert n_clusters//x[KEY].size(0) == self.num_clusters // n_global_crops
            
            teacher_cls_tokens = torch.cat((teacher_cls_tokens, self.teacher.dino_head.att(x[KEY])))
            ibot_teacher_patch_tokens = x["x_norm_patchtokens"]
            
            _dim = ibot_teacher_patch_tokens.shape[-1]

            if do_ibot:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head_1 = self.teacher.cluster_head(teacher_cls_tokens[:n_clusters])
                teacher_cls_tokens_after_head_2 = self.teacher.dino_head(teacher_cls_tokens[n_clusters:])
                if self.ibot_separate_head:
                    masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                        :n_masked_patches
                    ]
                else:
                    masked_teacher_patch_tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)[
                        :n_masked_patches
                    ]
            else:
                raise NotImplementedError

            if self.cfg.train.centering == "centering":
                raise NotImplementedError
                
            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_c = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head_1, teacher_temp=teacher_temp
                )
                
                teacher_dino_softmaxed_centered_g = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head_2, teacher_temp=teacher_temp
                )
                
                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return (teacher_dino_softmaxed_centered_c,teacher_dino_softmaxed_centered_g), masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        teacher_dino_softmaxed_centered = teacher_dino_softmaxed_centered[0].chunk(self.num_clusters) + teacher_dino_softmaxed_centered[1].chunk(n_global_crops)
        
        loss_dict = {}

        loss_accumulator = 0 
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        student_local_cls_tokens = student_local_backbone_output_dict[KEY]
        student_global_cls_tokens = student_global_backbone_output_dict[KEY]
        
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_patchtokens"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                student_global_masked_patch_tokens_after_head = self.student.dino_head(buffer_tensor_patch_tokens.unsqueeze(0)).squeeze(0)[:n_masked_patches]
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        teacher_global_base = torch.cat((teacher_dino_softmaxed_centered[self.num_clusters:][::-1]))
        
        if n_local_crops > 0:
            
            w = 1 / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            student_local_base = self.student.dino_head(student_local_cls_tokens,pool=True)
              
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_base.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_global_base.chunk(n_global_crops), 
            ) 
            dino_local_crops_loss = dino_local_crops_loss * w
            
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss
            
        loss_scales = 2

        if do_dino:
            
            w = 1 / (n_global_crops_loss_terms + n_local_crops_loss_terms)
           
            student_global_base = self.student.dino_head(student_global_cls_tokens,pool=True)
            
            teacher_center = torch.cat((teacher_dino_softmaxed_centered[:self.num_clusters])).repeat(n_global_crops,1)[t_mask.flatten(0,1)]
            attmask = (t_assign_map.flatten(2) > 0.5).bool()
        
            student_global_cls_tokens_after_head = self.student.cluster_head.att(student_global_cls_tokens,self.num_clusters,mask=attmask)[t_mask]
            student_global_cls_tokens_after_head = self.student.cluster_head(student_global_cls_tokens_after_head)
            
            dino_global_crops_loss = self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_center
                    ],  
                ) * n_global_crops + self.dino_loss(
                    student_output_list=[student_global_base],
                    teacher_out_softmaxed_centered_list=[
                        teacher_global_base
                    ],  
                ) * n_global_crops
            
            dino_global_crops_loss = dino_global_crops_loss * w

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss
            loss_dict["koleo_loss"] = dino_global_crops_loss * 0

        if do_ibot:
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss
        else:
            loss_dict["ibot_loss"] = torch.zeros(1).sum().to(student_global_base)
        return loss_accumulator, loss_dict
    
    def update_teacher(self, m):
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    mt.data = mt.data * m + ms.data * (1.0 - m)   
                    
    def train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.teacher.eval()

    def prepare_for_distributed_training(self):
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())


