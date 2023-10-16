# Copyright (c) SenseTime Research. All rights reserved.

from PIL import Image
import os
import torch
from tqdm import tqdm
from pti.pti_configs import paths_config, hyperparameters, global_config
from pti.pti_configs.denseclip_fpn_vit_b_640x640_80k import CONF, data_meta
from pti.training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from torchvision.utils import save_image
from .localitly_regulizer import Space_Regulizer, l2_loss
import numpy as np
import sys
sys.path.append('../criteria/human_parse/denseclip')
from denseclip import DenseCLIP
import pickle

import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

class EditCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def get_mask(self,gen_img_seg,ori_seg,edit_seg):

        seg_item=torch.unique(gen_img_seg)
        ori_mask=torch.zeros_like(ori_seg)
        edit_mask=torch.zeros_like(edit_seg)
        gen_ori_mask=torch.zeros_like(ori_seg)
        gen_edit_mask=torch.zeros_like(edit_seg)

        for item_idx in seg_item:
            if item_idx not in [1,2,3,4,5,6,21]:
                if item_idx in [0,15]:
                    continue
                else:
                    ori_mask=torch.where(ori_seg==item_idx,torch.ones_like(ori_mask),ori_mask)
                    gen_ori_mask=torch.where(gen_img_seg==item_idx,torch.ones_like(gen_ori_mask),gen_ori_mask)
            
            else:
                edit_mask=torch.where(edit_seg==item_idx,torch.ones_like(edit_mask),edit_mask)
                gen_edit_mask=torch.where(gen_img_seg==item_idx,torch.ones_like(gen_edit_mask),gen_edit_mask)
        
        edit_mask=torch.where(((edit_seg==0) * (ori_seg!=0)),torch.ones_like(edit_mask),edit_mask)
        edit_mask=torch.where(((edit_seg==0) * (ori_seg!=15)),torch.ones_like(edit_mask),edit_mask)# add
        edit_mask=torch.where(((edit_seg==15) * (ori_seg!=15)),torch.ones_like(edit_mask),edit_mask) 
        ori_mask=torch.where(((ori_seg==0)*(edit_mask==0)),torch.ones_like(ori_mask),ori_mask)
        ori_mask=torch.where(((ori_seg==15)*(edit_mask==0)),torch.ones_like(ori_mask),ori_mask)
        ori_mask=torch.where(((ori_seg==15)*(edit_mask==15)),torch.ones_like(ori_mask),ori_mask) # add


        return ori_mask,edit_mask

    def calcu_loss_seg(self,gen_img,gen_img_seg, ori_img,edit_img,ori_seg,edit_seg):
        loss=0.0
        ori_mask,edit_mask=self.get_mask(gen_img_seg,ori_seg,edit_seg)

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss(ori_mask*gen_img, ori_mask*ori_img)
            loss += (l2_loss_val) * hyperparameters.pt_l2_lambda    
        if hyperparameters.pt_lpips_lambda > 0:
        
            loss_lpips_1 = self.lpips_loss(ori_mask*gen_img, ori_mask*ori_img)
            loss_lpips_1 = torch.squeeze(loss_lpips_1)
            
            loss_lpips_2 = self.lpips_loss(edit_mask*gen_img, edit_mask*edit_img)
            loss_lpips_2 = torch.squeeze(loss_lpips_2)
            loss_lpips=loss_lpips_1+loss_lpips_2
           
            loss += loss_lpips * hyperparameters.pt_lpips_lambda
                
        return loss, l2_loss_val, loss_lpips
    
    def init_seg_model(self):
        self.paletee=get_palette(24)
        self.img_trans=transforms.Compose([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        cpk_file=paths_config.seg_model_path

        self.seg_model=DenseCLIP(**CONF)
        self.seg_model.load_state_dict(torch.load(cpk_file)['state_dict'])
        self.seg_model.cuda()
        self.seg_model.eval()

    def get_seg(self,img):
        img=self.img_trans(img)
        
        with torch.no_grad():
            seg_result_x=self.seg_model.simple_test(img,data_meta)

        return torch.from_numpy(np.array(seg_result_x))

    def get_paste_mask(self, ori_seg, edit_seg):
        ori_mask=torch.zeros_like(ori_seg)
        edit_mask=torch.zeros_like(edit_seg)

        for i in [  1,2,3,4,5,6,21]:
            edit_mask=torch.where(edit_seg==i,torch.ones_like(edit_mask),edit_mask)
        
        ori_mask=torch.where(edit_mask==0,torch.ones_like(ori_mask),ori_mask)

        return ori_mask,edit_mask
    
    def save_seg(self,img,image_name,path):
        s=Image.fromarray(np.asarray(img[0].detach().cpu(), dtype=np.uint8))
        s.putpalette(self.paletee)
        s.save(path)
    
    def train(self,start_idx=None,end_idx=None):

        test_list=os.listdir(paths_config.edit_file_dir)
        
        # TODO May Need modify according to different output
        image_name_list={}
        for filename in test_list:
            image_name=filename.split('+')[0]
            if image_name not in image_name_list:
                image_name_list[image_name]=[]
            
            new_name=image_name+'+'+filename.split('+')[1].split('.')[0]
            image_name_list[image_name].append(filename.split('.')[0])
            print(new_name)
            
        
        use_ball_holder = True
        self.init_seg_model()

        for fname, image in tqdm(self.data_loader):
            image_name_all= fname[0]
            print(fname)
            
            if image_name_all not in image_name_list:
                continue
            
            for image_name_full in image_name_list[image_name_all]:

                self.restart_training()
               
                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break

                w_pivot = None
                
                w_pivot = self.load_inversions_edit(paths_config.edit_w_dir, image_name_full)

                w_pivot = w_pivot.to(global_config.device)

                log_images_counter = 0
                real_images_batch = image.to(global_config.device)
                real_img_seg=self.get_seg(real_images_batch).to(global_config.device)
               
                for i in range(hyperparameters.max_pti_steps):

                    generated_images = self.forward(w_pivot)
                    gen_seg=self.get_seg(generated_images).to(global_config.device)
                    if i==0:
                        edit_img=generated_images.clone().detach().requires_grad_(True)
                        edit_img_seg=gen_seg.clone().detach()
                       
                    loss, l2_loss_val, loss_lpips=self.calcu_loss_seg(generated_images,gen_seg,real_images_batch,\
                                                                    edit_img,real_img_seg,edit_img_seg)
                    
                    if i == 0:
                        tmp1 = torch.clone(generated_images)
                    if i % 10 == 0:
                        print("pti loss: ", i, loss.data, loss_lpips.data)
                        
                    self.optimizer.zero_grad()

                    if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                        break

                    loss.backward()
                    self.optimizer.step()

                    use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                    global_config.training_step += 1
                    log_images_counter += 1

                # save output image
                print("ss!")
                
                save_image(generated_images.clamp(-1,1), f"{paths_config.experiments_output_dir}/{image_name_full}.png", normalize=True)
                
                tmp = torch.cat([real_images_batch.clamp(-1,1), tmp1.clamp(-1,1), generated_images.clamp(-1,1)], axis= 3)
                save_image(tmp, f"{paths_config.experiments_output_dir}/{image_name_full}_comp.png", normalize=True)
                
                self.image_counter += 1
