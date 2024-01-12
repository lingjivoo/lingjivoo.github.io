---
title: 3D Gaussian Splatting
author: Cheng Luo
date: 2024-01-11 15:46:00 +1100
categories: [Study]
tags: [Technique]
pin: true
---


# Code 
***

[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

Clone the github
···
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
···

## Setup
···
conda env create --file environment.yml
conda activate gaussian_splatting
···

**Notes**
Theu cuda version must $\geq 11.8$ 
And then you need to install submodules

···
pip install -q gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -q /content/gaussian-splatting/submodules/simple-knn
···
## Dataset
···
wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip
···

## Train
···
python train.py -s gaussian-splatting/tandt/train
···


## Eval
···
python render.py -m <path to trained model> 
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
···

··· <path to trained model> ··· may be ···outputs/xxxx···

---

作者：lingjivoo，github主页：[传送门](https://github.com/lingjivoo)
