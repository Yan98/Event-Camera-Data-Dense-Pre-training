from .logger import (MessageLogger, get_env_info, get_root_logger,
                     init_tb_logger, init_wandb_logger)
from .misc import scandir, MetricLogger
from .options import dict2str
import logging
import yaml
from yaml import CLoader as Loader
from flatten_dict import flatten, unflatten
import os 

__all__ = [
    'logger.py'
    'MessageLogger',
    'init_tb_logger',
    'init_wandb_logger',
    'get_root_logger',
    'get_env_info',
    'misc.py',
    'scandir',
    'options.py'
    'dict2str'
    'MetricLogger'
]

def init_loggers(opt,disable_print=False):
    log_file = opt['logger']["path"]
    logger = get_root_logger(
        logger_name=opt['logger']['name'], log_level=logging.INFO, log_file=log_file)
    if disable_print:
        return logger
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    return logger


def load(opt):
    opt = yaml.load(open(opt, mode='r'), Loader=Loader)
    base_opt = opt.get("base", None)
    if base_opt == None:
        return opt
    else:
        base_opt = flatten(load(base_opt))
        opt = flatten(opt)
        base_opt.update(opt)
        opt = unflatten(base_opt)
    return opt

def update_opt(opt, args):
    opt = load(opt)
    name = opt["name"]
    path = opt["exp_path"]
    
    os.makedirs(path,exist_ok=True)
    
    #set logger
    opt["logger"] = {}
    opt["logger"]["name"] = name
    opt["logger"]["path"] = os.path.join(path,"log_" + name)
    
    #checkpoint save path
    opt["train"]["save_path"] = os.path.join(path, "checkpoints")
    opt["train"]["sample"] = os.path.join(path, "sample")
    
    chk = os.path.join(opt["train"]["save_path"],"latest.pt")
    if os.path.exists(chk) and args.checkpoint is None and args.resume:
        args.checkpoint = chk
    return args, opt