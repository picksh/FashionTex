from builtins import print
from locale import T_FMT
import torch
import clip
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class CLIPLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CLIPLoss, self).__init__()
        self.opts=opts
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=opts.stylegan_size // 32)
        self.cos_loss2= torch.nn.CosineEmbeddingLoss()
   
    def new_clip_loss(self,ori_image,ori_text,tar_image,tar_text):
        ori_image = self.avg_pool(self.upsample(ori_image))
        tar_image = self.avg_pool(self.upsample(tar_image))

        emb_ori=self.model.encode_image(F.interpolate(ori_image,size=224)) # B 512
        emb_tar=self.model.encode_image(F.interpolate(tar_image,size=224))

        emb_ori_text=self.model.encode_text(clip.tokenize(ori_text[0]).cuda()) # B 512
        emb_tar_text=self.model.encode_text(clip.tokenize(tar_text[0]).cuda())
        change_text=torch.from_numpy(np.array(ori_text[1]==ori_text[0])).view(-1,1).repeat(1,emb_ori_text.shape[1]).cuda()
        emb_ori_text2=self.model.encode_text(clip.tokenize(ori_text[1]).cuda()) # B 512
        emb_ori_text2=torch.where(change_text==False,torch.zeros_like(emb_ori_text),emb_ori_text2)
        change_text=torch.from_numpy(np.array(tar_text[1])==np.array(tar_text[0])).view(-1,1).repeat(1,emb_tar_text.shape[1]).cuda()
        emb_tar_text2=self.model.encode_text(clip.tokenize(tar_text[1]).cuda())
        emb_tar_text2=torch.where(change_text==False,torch.zeros_like(emb_tar_text),emb_tar_text2)
        t_res=emb_ori-emb_ori_text-emb_ori_text2
        t_full=emb_tar_text+emb_tar_text2+t_res

        cos_target = torch.ones((emb_tar.shape[0])).float().cuda()
        similarity = self.cos_loss2(emb_tar, t_full, cos_target)
       
        return similarity
  


        
