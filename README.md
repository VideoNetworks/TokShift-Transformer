# TokShift-Transformer
This is official implementaion of paper "Token Shift Transformer for Video Classification". We achieve SOTA performance 80.40% on Kinetics-400 val.
<div align="center">
  <img src="demo/tokshift.PNG" width="800px"/>
</div>

- [Updates](#updates)
- [Model Zoo and Baselines](#model-zoo-and-baselines)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citing](#citing)
- [Acknowledgement](#Acknowledgement)



## Updates
### July 11, 2021
* Release this V1 version (the version used in paper) to public.
* we are preparing a V2 version which include the following modifications, will release within 1 week:
1. Directly decode video mp4 file during training/evaluation
2. Change to adopt standarlize timm code-base.
3. Performances are further improved than reported in paper version (average +0.5).


### April 22, 2021
* Add Train/Test guidline and Data perpariation
### April 16, 2021
* Publish TokShift Transformer for video content understanding

## Model Zoo and Baselines
| architecture | backbone |  pretrain |  Res & Frames | GFLOPs x views|  top1  |  config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| ViT (Video) | Base16 | ImgNet21k | 224 & 8 | 134.7 x 30 | 76.02 [`link`](https://drive.google.com/drive/folders/1Bj5tc9dQmmJbhouytPOIYJlvK6O2yFyE?usp=sharing)  |k400_vit_8x32_224.yml |
| TokShift | Base-16 | ImgNet21k | 224 & 8 | 134.7 x 30 | 77.28 [`link`](https://drive.google.com/drive/folders/1ty6OqhZpUxSzokXmP1IUFpze5A0Gihuu?usp=sharing) |k400_tokshift_div4_8x32_base_224.yml |
| TokShift (MR)| Base16 | ImgNet21k | 256 & 8 | 175.8 x 30 | 77.68 [`link`](https://drive.google.com/drive/folders/1xmW8VjYDdw996j17mHZhhjBnA2dJRj7a?usp=sharing)  |k400_tokshift_div4_8x32_base_256.yml |
| TokShift (HR)| Base16 | ImgNet21k | 384 & 8 | 394.7 x 30 | 78.14 [`link`](https://drive.google.com/drive/folders/1QNZWV9VUJuZdUzoU4NUtSZxFgLF3MONm?usp=sharing)  |k400_tokshift_div4_8x32_base_384.yml |
| TokShift | Base16 | ImgNet21k | 224 & 16 | 268.5 x 30 | 78.18 [`link`](https://drive.google.com/drive/folders/1w2XrDRLNJdDG1e6OczsxKEr6-LuwhVgv?usp=sharing) |k400_tokshift_div4_16x32_base_224.yml |
| TokShift-Large (HR)| Large16 | ImgNet21k | 384 & 8 | 1397.6 x 30 | 79.83 [`link`](https://drive.google.com/drive/folders/1tTXo5NzV4d9FTmnh40sUTXPoyUxWkq-I?usp=sharing)  |k400_tokshift_div4_8x32_large_384.yml |
| TokShift-Large (HR)| Large16 | ImgNet21k | 384 & 12 | 2096.4 x 30 | 80.40 [`link`](https://drive.google.com/drive/folders/1vuDcSZLgzsicJr9d1yKSm5089fiN6PX7?usp=sharing) |k400_tokshift_div4_12x32_large_384.yml |

Below is trainig log, we use 3 views evaluation (instead of 30 views) during validation for time-saving.
<div align="center">
  <img src="demo/trnlog.PNG" width="800px"/>
</div>

## Installation
* PyTorch >= 1.7, torchvision
* tensorboardx

## Quick Start
### Train
1. Download ImageNet-22k pretrained weights from [`Base16`](https://drive.google.com/file/d/1RMw1YO3hKQuK4hmcxqNZK_xi7LpxXVPp/view?usp=sharing) and [`Large16`](https://drive.google.com/file/d/12TkF_wFZn5JkpqjBE_CVmG3j8CTEas5K/view?usp=sharing).
2. Prepare Kinetics-400 dataset organized in the following structure, [`trainValTest`](https://drive.google.com/file/d/1i-NoXsyYVH4_D3M7iWInviVG41FCmiqf/view?usp=sharing)
```
k400
|_ frames331_train
|  |_ [category name 0]
|  |  |_ [video name 0]
|  |  |  |_ img_00001.jpg
|  |  |  |_ img_00002.jpg
|  |  |  |_ ...
|  |  |
|  |  |_ [video name 1]
|  |  |   |_ img_00001.jpg
|  |  |   |_ img_00002.jpg
|  |  |   |_ ...
|  |  |_ ...
|  |
|  |_ [category name 1]
|  |  |_ [video name 0]
|  |  |  |_ img_00001.jpg
|  |  |  |_ img_00002.jpg
|  |  |  |_ ...
|  |  |
|  |  |_ [video name 1]
|  |  |   |_ img_00001.jpg
|  |  |   |_ img_00002.jpg
|  |  |   |_ ...
|  |  |_ ...
|  |_ ...
|
|_ frames331_val
|  |_ [category name 0]
|  |  |_ [video name 0]
|  |  |  |_ img_00001.jpg
|  |  |  |_ img_00002.jpg
|  |  |  |_ ...
|  |  |
|  |  |_ [video name 1]
|  |  |   |_ img_00001.jpg
|  |  |   |_ img_00002.jpg
|  |  |   |_ ...
|  |  |_ ...
|  |
|  |_ [category name 1]
|  |  |_ [video name 0]
|  |  |  |_ img_00001.jpg
|  |  |  |_ img_00002.jpg
|  |  |  |_ ...
|  |  |
|  |  |_ [video name 1]
|  |  |   |_ img_00001.jpg
|  |  |   |_ img_00002.jpg
|  |  |   |_ ...
|  |  |_ ...
|  |_ ...
|
|_ trainValTest
   |_ train.txt
   |_ val.txt
```

3. Using train-script (train.sh) to train k400
```
#!/usr/bin/env python
import os

cmd = "python -u main_ddp_shift_v3.py \
		--multiprocessing-distributed --world-size 1 --rank 0 \
		--dist-ur tcp://127.0.0.1:23677 \
		--tune_from pretrain/ViT-L_16_Img21.npz \
		--cfg config/custom/kinetics400/k400_tokshift_div4_12x32_large_384.yml"
os.system(cmd)

```
### Test
Using test.sh (test.sh) to evaluate k400
```
#!/usr/bin/env python
import os
cmd = "python -u main_ddp_shift_v3.py \
        --multiprocessing-distributed --world-size 1 --rank 0 \
        --dist-ur tcp://127.0.0.1:23677 \
        --evaluate \
        --resume model_zoo/ViT-B_16_k400_dense_cls400_segs8x32_e18_lr0.1_B21_VAL224/best_vit_B8x32x224_k400.pth \
        --cfg config/custom/kinetics400/k400_vit_8x32_224.yml"
os.system(cmd)
```

## Contributors
VideoNet is written and maintained by [Dr. Hao Zhang](https://hzhang57.github.io/) and [Dr. Yanbin Hao](https://haoyanbin918.github.io/).

## Citing
If you find TokShift-xfmr is useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@article{tokshift2021,
  title={Token Shift Transformer for Video Classification},
  author={Hao Zhang, Yanbin Hao, Chong-Wah Ngo},
  journal={ACM Multimedia 2021},
}
```

## Acknowledgement
Thanks for the following Github projects:
- https://github.com/rwightman/pytorch-image-models
- https://github.com/jeonsworld/ViT-pytorch
- https://github.com/mit-han-lab/temporal-shift-module
- https://github.com/amdegroot/ssd.pytorch

