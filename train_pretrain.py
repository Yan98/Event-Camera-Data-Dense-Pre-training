#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')
import argparse
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from trainer import create_trainer
from pytorch_lightning.strategies import DDPStrategy
from data import create_dataloader
from data.ETartanAir_dataset import TartanairPretrainDataset
from model import create_model
from utils import MessageLogger, init_loggers, update_opt
from copy import deepcopy
from pytorch_lightning.profiler import SimpleProfiler
import os 
    
#Speed up training
cudnn.benchmark = True

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class data_prep(pl.LightningDataModule):
    DATASETS = dict(
        TartanairPretrainDataset=TartanairPretrainDataset,
        )
    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)
       
    def setup(self, stage: str):   
        opt = self.opt
        dataset_opt = deepcopy(opt["datasets"])
        
        train_opt = deepcopy(dataset_opt)
        self.train_opt = train_opt
        
        eval_opt = deepcopy(dataset_opt)
        self.eval_opt = eval_opt
        
        dataset_opt = AttrDict(**dataset_opt)
        crop_size = dataset_opt.get("crop_size",224) 
        aug_params = {'crop_size': [crop_size,crop_size]}
        
        self.train_dataset = self.DATASETS[dataset_opt["type"]](dataset_opt,True,aug_params) 
        
    def train_dataloader(self):
        if hasattr(self, "train_loader"):
            return self.train_loader
        opt = self.opt
        train_loader = create_dataloader(self.train_dataset,self.train_opt,opt['logger']['name'])
        self.train_loader = train_loader
        return train_loader

    
def main(args):

    args, opt = update_opt(args.opt,args)
    if "torch_home" in opt:
        os.environ['TORCH_HOME'] = opt["torch_home"]
    
    init_loggers(opt)
    msg_logger = MessageLogger(opt)

    model = create_model(opt["network"],opt['logger']['name']) 
    model = create_trainer(opt['train']['type'], opt['logger']['name'], {"model": model, "log" : msg_logger, "opt" : opt["train"], "checkpoint": args.checkpoint, "acce": args.acce})
    
    if opt.get("apex",False):
        kwargs = {"amp_backend":"apex", "amp_level":"O1"}
    else:
        kwargs = {}
    sync_batchnorm = opt['train'].get('sync_batchnorm', False)
    check_val_every_n_epoch = opt['train'].get('check_val_every_n_epoch',1)
    if args.debug:
        kwargs.update({"limit_train_batches":5})
    plt = pl.Trainer(max_epochs = opt["train"].get("early_stop_epoch", opt["train"]["epoch"]) - model.past_epoch, num_nodes=args.num_nodes, precision = opt.get("precision",32), gpus=args.gpus,strategy=DDPStrategy(find_unused_parameters=False),checkpoint_callback = False, logger = False, profiler = SimpleProfiler(), sync_batchnorm = sync_batchnorm, replace_sampler_ddp = False, check_val_every_n_epoch = check_val_every_n_epoch, **kwargs)
    plt.fit(model,data_prep(opt))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser() 
    parser.add_argument("--gpus", default = 1, type = int)
    parser.add_argument("--acce", default = "ddp", type = str)
    parser.add_argument("--num_nodes", default = 1, type = int)
    parser.add_argument("--checkpoint", default = None, type = str)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--opt', type=str, default = "", help='Path to option YAML file.')
    
    args = parser.parse_args()
    main(args)
    
