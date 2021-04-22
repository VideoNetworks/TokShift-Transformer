# TokShift-Transformer
This is official implementaion of paper "Token Shift Transformer for Video Classification". We achieve SOTA performance 80.40% on Kinetics-400 val.
<div align="center">
  <img src="demo/tokshift.PNG" width="600px"/>
</div>

- [Updates](#updates)
- [Model Zoo and Baselines](#model-zoo-and-baselines)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citing](#citing)
- [Acknowledgement](#Acknowledgement)





## Updates
### April 16, 2021
* Publish TokShift Transformer for video content understanding

## Model Zoo and Baselines
| architecture | backbone |  pretrain |  Res & Frames | GFLOPs x views|  top1  |  config |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | 
| ViT (Video) | Base16 | ImgNet21k | 224 & 8 | 134.7 x 30 | 76.02 [`link`](https://drive.google.com/drive/folders/1Bj5tc9dQmmJbhouytPOIYJlvK6O2yFyE?usp=sharing)  |k400_vit_8x32_224.yml |
| TokShift | Base-16 | ImgNet21k | 224 & 8 | 134.7 x 30 | 77.28 [`link`]() |k400_tokshift_div4_8x32_base_224.yml |
| TokShift (MR)| Base16 | ImgNet21k | 256 & 8 | 175.8 x 30 | 77.68 [`link`]()  |k400_tokshift_div4_8x32_base_256.yml |
| TokShift (HR)| Base16 | ImgNet21k | 384 & 8 | 394.7 x 30 | 78.14 [`link`]()  |k400_tokshift_div4_8x32_base_384.yml |
| TokShift | Base16 | ImgNet21k | 224 & 16 | 268.5 x 30 | 78.18 [`link`]() |k400_tokshift_div4_16x32_base_224.yml |
| TokShift-Large (HR)| Large16 | ImgNet21k | 384 & 8 | 1397.6 x 30 | 79.83 [`link`]()  |k400_tokshift_div4_8x32_large_384.yml |
| TokShift-Large (HR)| Large16 | ImgNet21k | 384 & 12 | 2096.4 x 30 | 80.40 [`link`]() |k400_tokshift_div4_12x32_large_384.yml |


## Installation


## Quick Start




## Citing
If you find TokShift-xfmr is useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@article{tokshift2021,
  title={Token Shift Transformer for Video Classification},
  author={Anonymous},
  journal={Under Review},
}
```

## Acknowledgement
Thanks for the following Github projects:
- https://github.com/rwightman/pytorch-image-models
- https://github.com/jeonsworld/ViT-pytorch
- https://github.com/mit-han-lab/temporal-shift-module
- https://github.com/amdegroot/ssd.pytorch

