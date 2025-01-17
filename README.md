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
## images
# model strcuture
![image](https://github.com/user-attachments/assets/05421640-e3ea-4b88-b489-624cd0b4e5c1)
# performance
![image](https://github.com/user-attachments/assets/b9f5f945-828e-4080-9858-eceec077a984)
![image](https://github.com/user-attachments/assets/f5ae24d3-10d9-4215-987f-3a02b6336f6c)

## Requirements(recommend to use conda )
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install numpy
pip install opencv-python
pip install pillow
```

## Dataset
The VED-8k dataset contains over 8,000 meticulously annotated visibility images. To access the dataset: Download from :link: https://pan.baidu.com/s/15YbE21Q5nqCT93txXM8sug?pwd=f1e7 code: f1e7 
Extract to ./data directory
Directory structure should be:
the datasets structure will be follows：
```bash
├── data
│   ├── train
│   ├── val
│   ├── test
```
## train
without fog_vision
```bash
python train.py
```
with fog_vision
```bash
python train_fogvision.py
```

