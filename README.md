# [TPAMI 2026] Breaking Barriers, Localizing Saliency: A Large-scale Benchmark and Baseline for Condition-Constrained Salient Object Detection
[Runmin Cong](https://scholar.google.cz/citations?user=-VrKJ0EAAAAJ&hl),
[Zhiyang Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=02BOfjcAAAAJ), 
[Hao Fang*](https://fanghaook.github.io/),
[Sam Kwong](https://scholar.google.com.hk/citations?hl=zh-CN&user=_PVI6EAAAAAJ) and
[Wei Zhang](https://scholar.google.com.hk/citations?hl=zh-CN&user=qCWuPHsAAAAJ)

[[`paper`](https://ieeexplore.ieee.org/document/11297835)] [[`BibTeX`](#CitingCSOD)]

## 🚩 Highlights:

---
- **CSOD**: We launch the new task **Condition-Constrained Salient Object Detection (CSOD)** with solutions from data and 
model dimensions, enabling intelligent systems to reliably address complex visual challenges in real/open environments. 
We also construct the large-scale benchmark **CSOD10K**—the first SOD dataset covering diverse constrained conditions, 
including **10,000 images, 3 constraint types, 8 real-world scenes, 101 object categories, and pixel-level annotations.**

<div align="center">
  <img src="Fig/example.png" width="90%" height="90%"/>
</div><br/>

<div align="center">
  <img src="Fig/dataset.png" width="90%" height="90%"/>
</div><br/>

- **SOTA Performance**: We propose a unified end-to-end framework **CSSAM** for the CSOD task. We design a **Scene Prior-Guided Adapter (SPGA)** 
to enable the foundation model to better adapt to downstream constrained scenes. We propose a **Hybrid Prompt Decoding Strategy (HPDS)** 
that effectively generates and integrates multiple types of prompts to achieve adaptation to the SOD task.

<div align="center">
  <img src="Fig/main.png" width="90%" height="90%"/>
</div><br/>

## 🛠️Environment Setup

---

### Requirements
- Python 3.9+
- Pytorch 2.0+ (we use the PyTorch 2.4.1)
- CUDA 12.1 or other version

### Installation


**Step 1**: Create a conda environment and activate it.

```shell
conda create -n cssam python=3.9 -y
conda activate cssam
```

**Step 2**: Install [PyTorch](https://pytorch.org/get-started/previous-versions/#v231). If you have experience with PyTorch and have already installed it, you can skip to the next section.

**Step 3**: Install other dependencies from requirements.txt
```shell
pip install -r requirements.txt
```

### Dataset
Please create a data folder in your working directory and put the CSOD10K dataset in it for training or testing. 
CSOD10K is divided into two parts, with 7503 images for training and 2497 images for testing.

    data
      ├── CSOD10K
      │   ├── class_list.txt
      │   ├── train
      │   │   ├── image
      │   │   │   ├── 00001.jpg
      │   │   │   ├── ...
      │   │   ├── mask
      │   │   │   ├── 00001.png
      │   │   │   ├── ...
      │   ├── test
      │   │   ├── image
      │   │   │   ├── 00003.jpg
      │   │   │   ├── ...
      │   │   ├── mask
      │   │   │   ├── 00003.png
      │   │   │   ├── ...
you can get our CSOD10K dataset in [Baidu Disk](https://pan.baidu.com/s/1JK2yDq0QNft0rkd-0IuS7A?pwd=447k) (pwd:447k) or [Google Drive](https://drive.google.com/file/d/16QXWPAkOBgX0IVhNTVuKsVYt_R8sf4Il/view?usp=drive_link).

### Download SAM2 model weights 

Download the pretrained model of the scale you need:
- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

Save them in ./checkpoints

## 🚀Train & Evaluate

---

### Train

To train the model(s) in the paper, run this command:

```shell
bash ./scripts/train.sh
```

We also provide simple instructions if you want to train the base or tiny version of the model

```shell
bash ./scripts/train_base.sh
```
or
```shell
bash ./scripts/train_tiny.sh
```

### Evaluate
To test a model, run this command:
```shell
bash ./scripts/eval.sh
```
## 📦 Model Zoo

---
Pre-trained weights for CSSAM variants are available for download:  

| Model   | Params (M) | $MAE$ | $F_{β}^{max}$ | $S_{m}$ | $E_{m}$ | Download Link                                                                                 |
|---------|------------|-------|---------------|---------|---------|-----------------------------------------------------------------------------------------------|
| CSSAM-T | 42.88      | 0.040 | 0.870         | 0.871   | 0.903   | [Google Drive](https://drive.google.com/file/d/1jZZOtPXjliJB5fPJS0k_tIu933AVViI4/view?usp=drive_link)                                                                                      |
| CSSAM-B | 85.26      | 0.035 | 0.887         | 0.886   | 0.916   | [Google Drive](https://drive.google.com/file/d/1Q469wmIigyHxxHdZ0REi68v2PKQ3Yv6b/view?usp=drive_link) |
| CSSAM-L | 230.08     | 0.028 | 0.907         | 0.902   | 0.931   | [Google Drive](https://drive.google.com/file/d/1lJbrFLnhOrMDtWVBnC2HFKFQc1hcgIDh/view?usp=drive_link) |

## <a name="CitingCSOD"></a>⭐ BibTeX

---
If you use CSOD in your research, please cite our paper:  
```BibTeX
@ARTICLE{11297835,
  author={Cong, Runmin and Chen, Zhiyang and Fang, Hao and Kwong, Sam and Zhang, Wei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Breaking Barriers, Localizing Saliency: A Large-scale Benchmark and Baseline for Condition-Constrained Salient Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Salient Object Detection;Constrained Conditions;Benchmark Dataset;Scene Prior;Hybrid Prompt},
  doi={10.1109/TPAMI.2025.3642893}}
```

## ☑️ Acknowledgement


This repository is implemented based on the [Segment Anything Model](https://github.com/facebookresearch/sam2). Thanks to them for their excellent work.



