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
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## Setup
```
conda env create --file environment.yml
conda activate gaussian_splatting
```

**Notes**
Theu cuda version must $$\geq 11.8$$ 
And then you need to install submodules

```
pip install -q gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -q /content/gaussian-splatting/submodules/simple-knn
```
## Dataset
```
wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip
```

## Train
```
python train.py -s gaussian-splatting/tandt/train
```


## Eval
```
python render.py -m <path to trained model> 
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset>
python metrics.py -m <path to pre-trained model>
```

``` <path to trained model> ``` may be ```outputs/xxxx```


## Code Explanation

### Train.py
model: GaussianModel (variable: gaussians)
dataset: Scene (variable: scene)
render: render (function: render, defined in gaussian_renderer)

```
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
```

```
    viewpoint_stack = None
```

```
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
```

```
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
```

Spherical harmonic lighting (SH coefficients)

---

作者：lingjivoo，github主页：[传送门](https://github.com/lingjivoo)
