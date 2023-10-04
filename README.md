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
pip install pytorch-lightning==1.4.2
pip install git+https://github.com/openai/CLIP.git
```

3. Install required packages for [DenseCLIP](https://github.com/raoyongming/DenseCLIP).
4. Download Pretrained StyleGAN-Human weight(stylegan_human_v2_1024.pkl) from  https://github.com/stylegan-human/StyleGAN-Human
5. Download Pretrained [IR-SE50](https://drive.google.com/file/d/1FS2V756j-4kWduGxfir55cMni5mZvBTv/view) model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
6. Download Pretrained [DenseCLIP](https://drive.google.com/file/d/1cHpWEC49qNhYAQRVrV8Ex1PJYIIB84TO/view?usp=sharing) weight.

Default path for pretrained weights is ./pretrained. You can change the path in mapper/options/train_options.py

## Prepare data

In this project, we use [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) dataset. We use [e4e](https://github.com/omertov/encoder4editing) to invert images into latent space.
1. Download [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) dataset.
2. In order to use the pre-trained StyleGAN-Human model, we should align images with [Aligned raw images](https://github.com/stylegan-human/StyleGAN-Human/tree/main#aligned-raw-images). Put the aligned images in data/data_split/aligned.
3. Invert aligned images: The simplest way is to follow [Invert real image with PTI](https://github.com/stylegan-human/StyleGAN-Human/tree/main#invert-real-image-with-pti) and we only need the output embedding "0.pt" in  'outputs/pti/'. (Since we only need the output of e4e, you can comment out the finetuning code to save time.）
4. Run the data processing script:
```bash 
bash data/process.sh
```
## Training
You can set the GPU number in run.sh. If you would like to change the data, weights, output path or other settings, you can find them in mapper/options/train_options.py.
```
bash run.sh
```

## Acknowledgements

This code is based on [StyleCLIP](https://github.com/orpatashnik/StyleCLIP) and [HairCLIP](https://github.com/wty-ustc/HairCLIP)