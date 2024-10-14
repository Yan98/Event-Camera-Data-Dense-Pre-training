# Event Camera Data Dense Pre-training

This repository contains the PyTorch code for our paper "Event Camera Data Dense Pre-training".

> [paper](./asset/paper.pdf) | [supp](./asset/supp.pdf) | [project page](https://yan98.github.io/ECDDP/)

**The code and dataset will come soon!**


## Introduction
This paper introduces a self-supervised learning framework designed for pre-training neural networks tailored to dense prediction tasks using event camera data. Our approach utilizes solely event data for training.

Transferring achievements from dense RGB pre-training  directly to event camera data yields subpar performance. This is attributed to the spatial sparsity inherent in an event image (converted from event data), where many pixels do not contain information. To mitigate this sparsity issue, we encode an event image into event patch features, automatically mine contextual similarity relationships among patches, group the patch features into distinctive contexts, and enforce context-to-context similarities to learn discriminative event features.

For training our framework, we curate a synthetic event camera dataset featuring diverse scene and motion patterns.
Transfer learning performance on downstream dense prediction tasks illustrates the superiority of our method over state-of-the-art approaches.

## Framework

<div align=center>
<img src="asset/model.png", width=600/>
</div>

## Requirement
- torch 2.2.1+cu118
- mmseg 
- pytorch-lightning 1.6.4
- timm 0.9.16
- kornia 0.7.1
- torch_scatter 2.1.2+pt22cu118
- opencv-python 4.9.0.80
- pillow 10.2.0
- albumentations 1.4.0
- ttach 0.0.3
- mmsegmentation 1.2.2

## How to get the dataset

Please refer to [`generate_data`](./generate_data).

## How to pretrain

```
python train_pretrain.py --opt ./config/pretrain/swin_small.yml --gpus #NUM_GPUS --num_nodes #NUM_NODES
```

We provide an example of pre-trained [swin-t/7](https://drive.google.com/file/d/12OOBZa1HupsI7-E-Ct8VhdK05O98yNoi/view?usp=sharing).

## How to finetune

Download the pre-trained model.

```
python3 train_seg.py --opt config/seg/swin_small.yml --gpus #NUM_GPUS --num_nodes #NUM_NODES #Please change dataset_path and pretrained_checkpoint in the config file
python eval_seg.py --checkpoint *.pt  --opt config/seg/swin_small.yml #Please set the checkpoint path

```


## Contact
If you have any questions relating to our work, do not hesitate to contact [me](mailto:yan.yang@anu.edu.au?subject=ECDDP).

## Acknowledgement
ECDDP is built using the awesome [tartanair_tools](https://github.com/castacks/tartanair_tools), [ess](https://github.com/uzh-rpg/ess), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [DinoV2](https://github.com/facebookresearch/dinov2), [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI), and [mae](https://github.com/facebookresearch/mae).

## Citation

```
@misc{yang2024eventcameradatadense,
      title={Event Camera Data Dense Pre-training}, 
      author={Yan Yang and Liyuan Pan and Liu Liu},
      year={2024},
      eprint={2311.11533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.11533}, 
}
```
