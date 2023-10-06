# 1. MSCA: Temporal Cross-attention for Action Recognition

- [1. MSCA: Temporal Cross-attention for Action Recognition](#1-msca-temporal-cross-attention-for-action-recognition)
  - [1.1. Citation](#11-citation)
  - [1.2. Acknowledgement](#12-acknowledgement)
  - [1.4. Implementation](#14-implementation)
  - [1.5. Weights](#15-weights)
  - [1.6. Prepare dataset](#16-prepare-dataset)
  - [1.7. train and val](#17-train-and-val)

This is an official repo of paper "Temporal Cross-attention for Action Recognition" at ACCV2022 Workshop on Vision Transformers: Theory and applications (VTTA-ACCV2022).

- [CVF open access](https://openaccess.thecvf.com/content/ACCV2022W/TCV/html/Hashiguchi_Temporal_Cross-attention_for_Action_Recognition_ACCVW_2022_paper.html)
- [PDF](https://openaccess.thecvf.com/content/ACCV2022W/TCV/papers/Hashiguchi_Temporal_Cross-attention_for_Action_Recognition_ACCVW_2022_paper.pdf)
- [slide](https://drive.google.com/file/d/1RRtuB8SKQ2KlpD7jByD4kyOaKoBu0bgj/view?usp=share_link)
- [LNCS](https://link.springer.com/chapter/10.1007/978-3-031-27066-6_20)
- [DOI:10.1007/978-3-031-27066-6_20](<https://doi.org/10.1007/978-3-031-27066-6_20>)
- [arXiv:2204.0045](https://arxiv.org/abs/2204.00452)

## 1.1. Citation

```BibTeX
@InProceedings{Hashiguchi_2022_ACCV,
    author    = {Hashiguchi, Ryota and Tamaki, Toru},
    title     = {Temporal Cross-attention for Action Recognition},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV) Workshops},
    month     = {December},
    year      = {2022},
    pages     = {276-288}
}
```

## 1.2. Acknowledgement

We thank for the author of TokenShift:

- <https://github.com/VideoNetworks/TokShift-Transformer>

## 1.4. Implementation

MSCA is build upon [TokenShift](https://github.com/VideoNetworks/TokShift-Transformer).

## 1.5. Weights

Download ImageNet-22k pretrained weights from [`Base16`](https://drive.google.com/file/d/1RMw1YO3hKQuK4hmcxqNZK_xi7LpxXVPp/view?usp=sharing).

## 1.6. Prepare dataset

Prepare Kinetics-400 dataset organized in the following structure.

Almost same with TokenShift, with slight modificaitons. See [config file](config/custom/kinetics400/k400_attentionshift_div4_8x32_base_224.yml).

```text
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

## 1.7. train and val

```bash
python main.py --tune_from pretrain/ViT-B_16_Img21.npz --cfg config/custom/kinetics400/k400_attentionshift_div4_8x32_base_224.yml
```
