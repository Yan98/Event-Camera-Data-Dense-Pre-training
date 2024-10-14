import torch
import time
import datetime
import numpy as np
from utils import MetricLogger
import torch.nn.functional as F
from .base_trainer import BaseTrainer
import os 

class TrainerSeg(BaseTrainer):
    
    def __init__(self, model, log,opt,checkpoint):
        super().__init__()
        
        self.model = model
        self.log = log
        self.opt = opt
        self.checkpoint = checkpoint
        self.automatic_optimization = False
        self.start_time  = None
        self.past_epoch = 0 
        os.makedirs(opt["save_path"],exist_ok=True)  

        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            self.past_epoch = checkpoint["current_epoch"]
            del checkpoint
            
        self.num_classes = opt.get("num_classes",11)
        self.loss = TaskLoss(num_classes=self.num_classes)
        self.max_iou = float("-inf")
        
    def training_step(self,data,idx):
        
        if self.current_epoch == 0 and idx == 0:
            self.start_time  = time.time()
        
        event = data["event_voxel"]
        label = data["label"]
        
        self.adjust_learning_rate(idx)    
        optimizer = self.optimizers()
        
        pred_main, pred_aux = self.model(event)
        loss_main = self.loss(pred_main,label)
        loss_aux = self.loss(pred_aux,label)
        
        optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss_main + 0.4 * loss_aux)
        optimizer.step()
        
        self.produce_log(loss_main, loss_aux, idx)
    
    def validation_step(self, data, idx):
        event = data["event_voxel"]
        label = data["label"]
        pred_main, pred_aux = self.model(event)
        
        loss = self.loss(pred_main,label)

        self.metric_logger.update(loss=loss.item())
        confusion = semseg_compute_confusion(pred_main.argmax(1),label,num_classes=self.num_classes).view(-1)
        for key in range(self.num_classes**2):
            self.metric_logger.meters["c"+ str(key)].update(confusion[key], n=1)
            
    def on_validation_epoch_start(self):
        self.metric_logger = MetricLogger()
        self.log.raw("Initialized metric_logger")
    
    def validation_epoch_end(self,outputs):
        self.metric_logger.synchronize_between_processes()
        
        
        loss = self.metric_logger.loss.global_avg
        eval_result = [self.metric_logger.__getattr__("c"+ str(k)).total for k in range(self.num_classes**2)]
        eval_result = torch.FloatTensor(eval_result).view(self.num_classes,self.num_classes)
        acc = semseg_accum_confusion_to_acc(eval_result).item()
        iou = (semseg_accum_confusion_to_iou(eval_result)[0]).item()

        
        if self.trainer.is_global_zero and self.trainer.num_gpus != 0:
            
            if self.start_time != None and iou > self.max_iou:
                 self.save("best")
                 self.max_iou = max(self.max_iou,iou)
            
            self.log.raw(f"loss: {loss}, acc: {acc}, iou: {iou} | max_iou: {self.max_iou}")
            
            self.log.save_eval(
                self.get_current_epoch,
                {"loss":loss,
                 "acc": acc,
                 "iou": iou,
                 "max_iou": self.max_iou,
                    }
                )
        
    def produce_log(self, loss_main, loss_aux, idx):
        
        loss_main = self.all_gather(loss_main).mean().item()
        loss_aux = self.all_gather(loss_aux).mean().item()
        
        if self.trainer.is_global_zero and idx % 100 == 0:
            
            len_loader = self.num_training_steps
            
            batches_done = self.current_epoch  * len_loader + idx + 1
            batches_left = self.trainer.max_epochs * len_loader - batches_done
            time_left    = datetime.timedelta(seconds = batches_left * (time.time() - self.start_time) / batches_done)
            lr = self.optimizers().param_groups[0]['lr']
            self.log(
                {"current_epoch": self.get_current_epoch,
                 "max_epochs": self.trainer.max_epochs + self.past_epoch,  
                 "idx": idx,
                 "len_loader":len_loader,
                 "time_left": time_left,
                 "loss_main": loss_main,
                 "loss_aux": loss_aux,
                 "lr": lr,
                    }
                )
            
            self.log.save_train(
                self.current_epoch + self.past_epoch,
                idx,
                { "loss_main": loss_main, 
                   "loss_aux": loss_aux,  
                    }
                )
        
    def configure_optimizers(self):
            
        self.opt["lr"] = self.trainer.num_gpus * self.trainer.num_nodes * self.num_batch_size / 256 * self.opt["base_lr"] 
        self.log.raw(f"learning_rate: {self.opt['lr']}")
         
        optimizer = torch.optim.AdamW(
                                    self.model.parameters(),
                                    lr = self.opt["lr"],
                                    weight_decay = self.opt["weight_decay"]
                                    )

        if self.checkpoint != None:
            checkpoint = torch.load(self.checkpoint,map_location="cpu")
            key = "model" if "model" in checkpoint else "checkpoint"
            msg = self.model.load_state_dict(checkpoint[key],strict=False)
            self.log.raw(msg)
            optimizer.load_state_dict(checkpoint["optimizer"])
            del checkpoint
            self.log.raw(f"Load checkpoint: {self.checkpoint}")
          
        return optimizer
    
class BinaryDiceLoss(torch.nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    result = result.scatter_(1, input, 1)

    return result

class DiceLoss(torch.nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, num_classes=13, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        mask = target != self.ignore_index
        target = target * mask
        target = make_one_hot(torch.unsqueeze(target, 1), self.num_classes)
        target = target * mask.unsqueeze(1)

        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        predict = predict * mask.unsqueeze(1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]      

class TaskLoss(torch.nn.Module):
    def __init__(self, losses=['cross_entropy', 'dice'], gamma=2.0, num_classes=11, alpha=None, weight=None, ignore_index=255, reduction='mean'):
        super(TaskLoss, self).__init__()
        self.losses = losses
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=self.ignore_index)
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index) 

    def forward(self, predict, target):
        if predict.size(-1) != target.size(-1):
            target = torch.nn.functional.interpolate(target.float().unsqueeze(1),size=(predict.size(2),predict.size(3))).long().squeeze(1)

        total_loss = 0
        if 'dice' in self.losses:
            total_loss += self.dice_loss(predict, target)
        if 'cross_entropy' in self.losses:
            total_loss += self.ce_loss(predict, target)

        return total_loss

def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes=11, ignore_label=255):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y_lbl), 'Inputs must be torch tensors'
    assert y_lbl.device == y_hat_lbl.device, 'Input tensors have different device placement'

    assert y_hat_lbl.dim() == 3 or y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1
    assert y_lbl.dim() == 3 or y_lbl.dim() == 4 and y_lbl.shape[1] == 1
    if y_hat_lbl.dim() == 4:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y_lbl.dim() == 4:
        y_lbl = y_lbl.squeeze(1)

    mask = y_lbl != ignore_label
    y_hat_lbl = y_hat_lbl[mask]
    y_lbl = y_lbl[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + num_classes * y_lbl
    bincount_2d = torch.bincount(x.long(), minlength=num_classes ** 2)
    assert bincount_2d.numel() == num_classes ** 2, 'Internal error'
    conf = bincount_2d.view((num_classes, num_classes)).long()
    return conf

def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = 100 * diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class

def semseg_accum_confusion_to_acc(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    acc = 100 * diag.sum() / (conf.sum(dim=1).sum()).clamp(min=1e-12)
    return acc
        