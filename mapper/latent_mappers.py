import torch
from torch import nn
from torch.nn import Module
import clip
from models.stylegan2.model import EqualLinear, PixelNorm
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ModulationModule(Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding, cut_flag):
        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag == 1:
            return x
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out

class SubMapper(Module):
    def __init__(self, opts, layernum):
        super(SubMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(5)])

    def forward(self, x, embedding, cut_flag=0):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
        	x = modulation_module(x, embedding, cut_flag)        
        return x

class Mapper(Module): 
    def __init__(self, opts):
        super(Mapper, self).__init__()
        self.opts = opts
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.style_cut_flag = 0
        self.color_cut_flag = 0

        self.medium_mapping = SubMapper(opts, 4)
        self.second_type_mapper=SubMapper(opts,4)
        
        self.fine_mapping = SubMapper(opts, 10)
        self.second_color_mapper=SubMapper(opts, 10)


    def gen_image_embedding(self, img_tensor, clip_model):
        masked_generated = self.face_pool(img_tensor)
        masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)
        return clip_model.encode_image(masked_generated_renormed)

    def forward(self, x, type_text_up,  color_tensor_up, color_tensor_low=None, type_text_low=None):
       
        type_emb_up = self.clip_model.encode_text(type_text_up).unsqueeze(1).repeat(1, 18, 1).detach()
        type_emb_low = self.clip_model.encode_text(type_text_low).unsqueeze(1).repeat(1, 18, 1).detach()
       
        texture_emb_up = self.gen_image_embedding(color_tensor_up, self.clip_model).unsqueeze(1).repeat(1, 18, 1).detach()
        texture_emb_low=self.gen_image_embedding(color_tensor_low,self.clip_model).unsqueeze(1).repeat(1, 18, 1).detach()

        x_coarse=x[:,:4,:]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse, type_emb_up[:, :4, :], cut_flag=self.style_cut_flag)
        else:
            x_coarse = torch.zeros_like(x_coarse)

        if not self.opts.no_medium_mapper:
            x_medium1 = self.medium_mapping(x_medium, type_emb_up[:, 4:8, :], cut_flag=self.style_cut_flag)
            x_medium2 = self.second_type_mapper(x_medium, type_emb_low[:, 4:8, :], cut_flag=self.style_cut_flag)
            x_medium= 0.5*x_medium1+0.5*x_medium2
        else:
            x_medium = torch.zeros_like(x_medium)

        if not self.opts.no_fine_mapper:
            x_fine1 = self.fine_mapping(x_fine, texture_emb_up[:, 8:, :], cut_flag=self.color_cut_flag)
            x_fine2= self.second_color_mapper(x_fine,texture_emb_low[:, 8:, :], cut_flag=self.color_cut_flag)
            x_fine=0.5*x_fine1+0.5*x_fine2
        else:
            x_fine = torch.zeros_like(x_fine)
        
        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out