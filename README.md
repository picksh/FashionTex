# FashionTex
The official implementation of SIGGRAPH 2023 conference paper, FashionTex: Controllable Virtual Try-on with Text and Texture.
(https://arxiv.org/abs/2305.04451)

## TODO:

- [x] Training Code
- [ ] Data Processing Script
- [ ] Test Code
- [ ] ID Recovery Module

## Requirement
1. Create a conda virtual environment and activate it:
```shell
conda create -n fashiontex python=3.8
conda activate fashiontex
```

2. Install required packages:
```shell
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

3. Install required packages for [DenseCLIP](https://github.com/raoyongming/DenseCLIP).
4. Download Pretrained StyleGAN-Human weight(stylegan_human_v2_1024.pkl) from  https://github.com/stylegan-human/StyleGAN-Human
5. Download Pretrained [IR-SE50](https://drive.google.com/file/d/1FS2V756j-4kWduGxfir55cMni5mZvBTv/view) model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
6. Download Pretrained [DenseCLIP]() weight.

Default path for pretrained weights is ./pretrained. You can change the path in mapper/options/train_options.py

## Prepare data

In this project, we use [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) dataset. We use [e4e](https://github.com/omertov/encoder4editing) to invert images into latent space.




## Acknowledgements

This code is based on [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and [HairCLIP](https://github.com/wty-ustc/HairCLIP)