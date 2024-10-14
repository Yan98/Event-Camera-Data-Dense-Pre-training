#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import argparse
from model import create_model
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from trainer import create_trainer
from pytorch_lightning.plugins import DDPPlugin
from data import create_dataloader
from data.DESC_dataset import DatasetProvider 
from utils import MessageLogger, init_loggers, update_opt
from copy import deepcopy
from pytorch_lightning.profiler import SimpleProfiler
import os 
from model.encoderdecoder_model import EncoderDecoder
cudnn.benchmark = True

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DESCdata_prep(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)

    def setup(self, stage: str):
        opt = self.opt
        dataset_opt = deepcopy(opt["datasets"])

        train_opt = deepcopy(dataset_opt)
        train_opt['phase'] = "train"
        self.train_opt = train_opt

        eval_opt = deepcopy(dataset_opt)
        eval_opt['phase'] = "eval"
        self.eval_opt = eval_opt

        self.train_dataset = DatasetProvider(**dataset_opt,mode='train').get_dataset()
        self.val_dataset = DatasetProvider(**dataset_opt,mode='val').get_dataset()

    def train_dataloader(self):
        if hasattr(self, "train_loader"):
            return self.train_loader
        opt = self.opt
        train_loader = create_dataloader(self.train_dataset,self.train_opt,opt['logger']['name'])
        self.train_loader = train_loader
        return train_loader

    def val_dataloader(self):
        if hasattr(self, "eval_loader"):
            return self.eval_loader
        opt = self.opt
        eval_loader = create_dataloader(self.val_dataset,self.eval_opt,opt['logger']['name'],ddp_sampler=False)
        
        self.eval_loader = eval_loader
        return eval_loader

def main(args):

    args, opt = update_opt(args.opt,args)
    if "torch_home" in opt:
        os.environ['TORCH_HOME'] = opt["torch_home"]
    
    init_loggers(opt)
    msg_logger = MessageLogger(opt)

    
    backbone = create_model(opt["network"],opt['logger']['name'])
    
    model = EncoderDecoder(
            backbone,
            opt["head_main"],
            opt["head_aux"]
            )
    model = create_trainer(opt['train']['type'], opt['logger']['name'], {"model": model, "log" : msg_logger, "opt" : opt["train"], "checkpoint": args.checkpoint})
    
    if opt.get("apex",False):
        kwargs = {"amp_backend":"apex", "amp_level":"O1"}
    else:
        kwargs = {}
    sync_batchnorm = opt['train'].get('sync_batchnorm', True)
    check_val_every_n_epoch = opt['train'].get('check_val_every_n_epoch',1)
    plt = pl.Trainer(max_epochs = opt["train"].get("early_stop_epoch", opt["train"]["epoch"]) - model.past_epoch, num_nodes=args.num_nodes, precision = opt.get("precision",32), gpus=args.gpus,strategy=DDPPlugin(find_unused_parameters=False),checkpoint_callback = False, logger = False, profiler = SimpleProfiler(), sync_batchnorm = sync_batchnorm, replace_sampler_ddp = False, check_val_every_n_epoch = check_val_every_n_epoch, **kwargs)
    
    plt.fit(model,DESCdata_prep(opt))
    

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
    
