# Fog-Vision: Accurate Image-based Visibility Estimation

## Introduction
Fog-Vision is a novel end-to-end framework for image-based visibility estimation, featuring a dual-branch architecture with Automatic Seed Region Segmentation (ASRS) and Fog Feature Guided Attention (FFGA) modules. This repository contains the official implementation of the paper "Fog-Vision: Accurate Image-based Visibility Estimation with Dual-branch Neural Networks".

## Key Features
- VED-8k Dataset: A large-scale dataset with 8,000+ annotated visibility images
- Dual-branch Architecture:
  - Primary Branch: Automatic Seed Region Segmentation (ASRS)
  - Secondary Branch: Flexible backbone support (CNNs, Transformers)
- Fog Feature Guided Attention (FFGA) module
- State-of-the-art Performance:
  - Accuracy: 97.7%
  - Precision: 97.2%
  - Recall: 95.2%

## Requirements(recommend to use conda )
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy
pip install opencv-python
pip install pillow
```bash

## Dataset
The VED-8k dataset contains over 8,000 meticulously annotated visibility images. To access the dataset: Download from [dataset link]
Extract to ./data directory
Directory structure should be:
the datasets structure will be follows：
├── data
│   ├── train
│   ├── val
│   ├── test
## Dataset
