#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import argparse
from model import create_model
import torch.backends.cudnn as cudnn
from data.DESC_dataset import DatasetProvider 
from utils import update_opt
from copy import deepcopy
from model.encoderdecoder_model import EncoderDecoder
cudnn.benchmark = True
import ttach as tta
try:
    from mmseg.ops import resize
except:
    from mmseg.models.utils import resize
import torch
from trainer.seg_trainer import semseg_compute_confusion, semseg_accum_confusion_to_iou 

def forward(self, img):
    """Forward function for training.
    Args:
        img (Tensor): Input images.
    Returns:
        dict[str, Tensor]: a dictionary of loss components
    """
    x = self.extract_feat(img)
    x_main = self._decode_head_forward(x)
    x_main = resize(
        input=x_main,
        size=img.shape[2:],
        mode='bilinear',
        align_corners=self.align_corners
        )


    return x_main

EncoderDecoder.forward = forward

def semseg_accum_confusion_to_macc(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    acc = 100 * diag / conf.sum(dim=1).clamp(min=1e-12)
    return acc.mean()

@torch.no_grad()
def main(args):

    args, opt = update_opt(args.opt,args)    
    backbone = create_model(opt["network"],opt['logger']['name'])
    
    model = EncoderDecoder(
            backbone,
            opt["head_main"],
            opt["head_aux"]
            )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")['model']
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()
    
    transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Scale(scales=[1,1.5,2],interpolation="bilinear", align_corners = True),
            ]
        )

    model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')    
    dataset_opt = deepcopy(opt["datasets"])
    eval_opt = deepcopy(dataset_opt)
    eval_opt['phase'] = "eval"
    val_dataset = DatasetProvider(**dataset_opt,mode='val').get_dataset()
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=2, drop_last=False, pin_memory=True, persistent_workers = True, shuffle=False)

    confusion = 0
    for data in val_dataset:
        
        event = data["event_voxel"].cuda()
        label = data["label"].cuda()

        pred_main = model(event)
        confusion += semseg_compute_confusion(pred_main.argmax(1),label).float()
    acc = semseg_accum_confusion_to_macc(confusion).item()
    iou = (semseg_accum_confusion_to_iou(confusion)[0]).item()

    print(args.opt,args.checkpoint)
    print(acc, iou)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--checkpoint", default = None, type = str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--opt', type=str, default = "", help='Path to option YAML file.')
    
    args = parser.parse_args()
    main(args)
    
