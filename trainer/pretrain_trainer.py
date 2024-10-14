#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
import datetime
import os
import math
import numpy as np
from .base_trainer import BaseTrainer

class TrainerPretrain(BaseTrainer):
    
    def __init__(self, model, log,opt,checkpoint, acce):
        super().__init__()
        self.model= model
        self.log = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = False
        self.start_time  = None
        self.past_epoch = 0 
        self.acce = acce
        os.makedirs(opt["save_path"],exist_ok=True)
        
        self.log.raw(f"checkpoint: {self.checkpoint}")
        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            self.past_epoch = checkpoint["current_epoch"]
            del checkpoint 

        warmup_teacher_temp = self.opt.get("warmup_teacher_temp",0.04)
        teacher_temp = self.opt.get("teacher_temp",0.07)
        teacher_temp_warmup_epochs = self.opt.get("teacher_temp_warmup_epochs",30)
         
        self.log.raw(f"warmup_teacher_temp: {warmup_teacher_temp} teacher_temp: {teacher_temp}  teacher_temp_warmup_epochs: {teacher_temp_warmup_epochs}")
        self.teacher_temps = np.concatenate(
            (np.linspace(warmup_teacher_temp, teacher_temp,
                         teacher_temp_warmup_epochs),
             np.ones(self.opt["epoch"] - teacher_temp_warmup_epochs) * teacher_temp)
            )
         
    def training_step(self,event,idx):
        
        warmup_cluster_epoch = self.opt.get("warmup_cluster_epoch",0)
        if warmup_cluster_epoch > self.get_current_epoch:
            self.model.warmup_weight = self.get_current_epoch / warmup_cluster_epoch + idx / self.num_training_steps
        else:
            self.model.warmup_weight = 1
        
        m = self.adjust_moco_momentum(idx)
        temp = self.adjust_teacher_temp()
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()

        self.adjust_learning_rate(idx)
        self.adjust_weight_decay(idx)
        
        delay_epoch = self.opt.get("delay_epoch", None)
        optimizer = self.optimizers()
        loss, loss_dict = self.model(event,m,temp)
        self.manual_backward(loss)
                

        if delay_epoch is not None and delay_epoch > self.get_current_epoch:
            for n, p in self.model.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
            
        self.produce_log(loss_dict, m, temp,  idx)  
    
    def adjust_teacher_temp(self):
        return self.teacher_temps[self.get_current_epoch]

    def adjust_moco_momentum(self, idx):
        epoch = self.get_current_epoch + idx / self.num_training_steps
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.opt["epoch"])) * (1. - self.opt["moco_m"])
        return m
    
    def adjust_weight_decay(self,idx):
        """Decays the learning rate with half-cycle cosine after warmup"""
        optimizer = self.optimizers()
        if "weight_decay_end" not in self.opt:
            return 
        weight_decay_end = self.opt["weight_decay_end"]
        min_weight_decay = self.opt["weight_decay"]
        epochs = self.opt["epoch"]
        epoch = self.get_current_epoch % epochs + idx / self.num_training_steps
        
        weight_decay = weight_decay_end + (min_weight_decay - weight_decay_end) * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
            
        for param_group in optimizer.param_groups:
            if "wd_multiplier" in param_group:
                param_group["weight_decay"] = weight_decay * param_group["wd_multiplier"]
            else:
                param_group["weight_decay"] = weight_decay
    def produce_log(self,loss_dict,m,temp,idx):
        
        l=loss_dict["dino_local_crops_loss"]
        g=loss_dict["dino_global_crops_loss"]
        k=loss_dict["koleo_loss"]
        i=loss_dict["ibot_loss"]
        l = self.all_gather(l).mean().item()
        g =  self.all_gather(g).mean().item()
        k =  self.all_gather(k).mean().item()
        i =  self.all_gather(i).mean().item()
        
        if self.trainer.is_global_zero and idx % 100 == 0:
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
            
            lr = self.optimizers().param_groups[0]['lr']
            wd = self.optimizers().param_groups[0]['weight_decay']
            self.log({"current_epoch": self.get_current_epoch,
                 "max_epochs": self.trainer.max_epochs + self.past_epoch,  
                 "idx": idx,
                 "len_loader":len_loader,
                 "time_left": time_left,
                 "l": l,
                 "g": g,
                 "k":k,
                 "i":i,
                 "m":m,
                 "temp":temp,
                 "lr": lr,
                 "wd": wd
                    })
            
            self.log.save_train(
                self.get_current_epoch,
                idx,
                { "l": l,
                 "g": g,
                 "k":k,
                 "i":i,
                 "lr": lr,
                  }
                )       

    @staticmethod
    def params_groups(model,patch_embed_lr_scale):
        regularized = []
        not_regularized = []
        patch_embed = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "patch_embed" in name:
                patch_embed.append(param)
            elif name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized, 'wd_multiplier':1}, {'params': not_regularized, 'wd_multiplier': 0}, {'params':patch_embed, 'wd_multiplier':1,'lr_scale':patch_embed_lr_scale}]
     

    def configure_optimizers(self):
        
        batch_size =  int(self.trainer.num_gpus  * self.trainer.num_nodes * self.num_batch_size)
        self.log.raw(f"batch_size: {batch_size}")
        
        
        self.opt["lr"] = self.opt["base_lr"]  * batch_size / 256
        
        params_groups = self.params_groups(self.model.student,self.opt["patch_embed_lr_scale"])
      
        optimizer = torch.optim.AdamW(
                            params_groups,
                            lr = self.opt["lr"],
                            weight_decay=self.opt["weight_decay"],
                            betas = (0.9, 0.999),
            )
        
        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.past_epoch = checkpoint["current_epoch"]
            del checkpoint
        torch.cuda.empty_cache()
        return optimizer                     
       
        