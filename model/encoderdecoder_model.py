import torch.nn as nn
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models import builder
from copy import deepcopy
try:
    from mmseg.ops import resize
except:
    from mmseg.models.utils import resize
decode_head_default=dict(
        type='UPerHead',
        in_channels=[384, 384, 384, 384],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512, 
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        )

auxiliary_head_default=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        )


class EncoderDecoder(BaseSegmentor):
    
    def __init__(self,
                 backbone,
                 decode_head_config = None,
                 auxiliary_head_config = None,
                 init_cfg = None,):
        super(EncoderDecoder, self).__init__(init_cfg)
        
        self.backbone = backbone
        
        decode_head = deepcopy(decode_head_default)
        decode_head.update(decode_head_config)
        
        auxiliary_head = deepcopy(auxiliary_head_default)
        auxiliary_head.update(auxiliary_head_config)
        
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = {}
        self.test_cfg = {}
        
    def _init_decode_head(self, decode_head):
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head):
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def _decode_head_forward(self, x):
        x = self.decode_head(x)
        return x

    def _auxiliary_head_forward(self, x):
        x = self.auxiliary_head(x)
        return x

    def forward(self, img):
        x = self.extract_feat(img)
        x_main = self._decode_head_forward(x)
        x_aux = self._auxiliary_head_forward(x)
        x_main = resize(
            input=x_main,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
            )
        
        x_aux = resize(
            input=x_aux,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners
            )
        
        return x_main, x_aux
    
    def _forward(self, x):
        raise SystemExit
        
    def loss(self,x):
        raise SystemExit
        
    def predict(self,x):
        raise SystemExit
    
    def aug_test(self, imgs, img_metas, **kwargs):
        pass
    
    def encode_decode(self, img, img_metas):
        pass
     
    def simple_test(self, img, img_meta, **kwargs):
        pass
    
    def forward_train(self, imgs, img_metas, **kwargs):
        pass    
