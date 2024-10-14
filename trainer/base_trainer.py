#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import pytorch_lightning as pl
import os
import math

class BaseTrainer(pl.LightningModule):
    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(dataset)
        
    @property
    def num_batch_size(self) -> int:
        dataset =  self.trainer._data_connector._train_dataloader_source.dataloader()
        return dataset.batch_size    
    
    @property
    def get_current_epoch(self) -> int:
        return self.past_epoch + self.current_epoch

        
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if "max_grad_norm" in self.opt:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt["max_grad_norm"])
            
    def adjust_learning_rate(self,idx):
        """Decays the learning rate with half-cycle cosine after warmup"""
        optimizer = self.optimizers()
        warmup_epochs = self.opt["warmup_epoch"]
        lr = self.opt["lr"]
        minlr = self.opt.get("min_lr", 0.0)
        epochs = self.opt["epoch"]
        epoch = self.get_current_epoch % epochs + idx / self.num_training_steps
        
        if epoch < warmup_epochs:
            lr = minlr + (lr - minlr) * epoch / warmup_epochs
        else:
            lr = minlr + (lr - minlr) * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
            
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
    
    def training_epoch_end(self, training_step_outputs):
        if "save_every" in self.opt and self.trainer.is_global_zero and (self.get_current_epoch + 1) % self.opt["save_every"] == 0:
            self.save() 
            self.log.raw("Model saved")
        if self.trainer.is_global_zero:
            self.save("latest")

    def save(self,name=None):
        if name == None:
            output_path = os.path.join(self.opt["save_path"], f"{self.get_current_epoch + 1}.pt")
        else:
            output_path = os.path.join(self.opt["save_path"], f"{name}.pt")
        torch.save(
            {
             "model": self.model.state_dict(),
             "optimizer": self.optimizers().state_dict(),
             "current_epoch": self.get_current_epoch + 1,
             }
            , output_path)
        
        