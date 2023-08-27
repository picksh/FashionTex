import torch
from torch import nn
import numpy as np
import lpips
from torchvision import transforms
from criteria.human_parse.denseclip.denseclip import DenseCLIP
from criteria.human_parse.configs.denseclip_fpn_vit_b_640x640_80k import CONF,data_meta
from criteria.human_parse.style_loss import StyleLoss

class SegLoss(nn.Module):
    def __init__(self,opts, seg_model='clip'):
        super(SegLoss, self).__init__()
        self.seg_model=seg_model
        self.style_loss=StyleLoss()
        if seg_model=='clip':
            self.model=DenseCLIP(**CONF)
            self.model.load_state_dict(torch.load(opts.seg_model_path)['state_dict'])
            self.model.cuda()
            self.model.eval()
        elif seg_model=='SCHP':
            self.model=SCHPmodel('atr')
        
        self.opts=opts
        self.bg_mask_l2_loss = nn.MSELoss()
        self.color_l1_loss = nn.L1Loss()
        self.percept = lpips.LPIPS(net='vgg').cuda()
        self.M = torch.tensor([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])

    # cal lab written by liuqk
    def f(self, input):
        output = input * 1
        mask = input > 0.008856
        output[mask] = torch.pow(input[mask], 1 / 3)
        output[~mask] = 7.787 * input[~mask] + 0.137931
        return output

    def rgb2xyz(self, input):
        assert input.size(1) == 3
        M_tmp = self.M.to(input.device).unsqueeze(0)
        M_tmp = M_tmp.repeat(input.size(0), 1, 1)  # BxCxC
        output = torch.einsum('bnc,bchw->bnhw', M_tmp, input)  # BxCxHxW
        M_tmp = M_tmp.sum(dim=2, keepdim=True)  # BxCx1
        M_tmp = M_tmp.unsqueeze(3)  # BxCx1x1
        return output / M_tmp

    def xyz2lab(self, input):
        assert input.size(1) == 3
        output = input * 1
        xyz_f = self.f(input)
        # compute l
        mask = input[:, 1, :, :] > 0.008856
        output[:, 0, :, :][mask] = 116 * xyz_f[:, 1, :, :][mask] - 16
        output[:, 0, :, :][~mask] = 903.3 * input[:, 1, :, :][~mask]
        # compute a
        output[:, 1, :, :] = 500 * (xyz_f[:, 0, :, :] - xyz_f[:, 1, :, :])
        # compute b
        output[:, 2, :, :] = 200 * (xyz_f[:, 1, :, :] - xyz_f[:, 2, :, :])
        return output

    def seg_with_text(self,seg_result,text,seg_otherwise=False):

        mask=torch.zeros_like(seg_result)
        #if 'outer' in text:
        #    text.append('top')
        if 'top' in text and 'outer' not in text:
            text.insert(0,'outer')
        for class_name in text:
            choose_text=class_name
            if self.seg_model=='clip':
                class_id=CONF['class_names'].index(class_name)
            else:
                class_id=self.model.label.index(class_name)
            mask=torch.where(seg_result==class_id,torch.ones_like(mask),mask) # 1 1024 512
            if mask.max()==1:
                break
        if seg_otherwise:
            otherwise_mask=torch.zeros_like(seg_result)
            otherwise_mask=torch.where(seg_result!=class_id ,torch.ones_like(otherwise_mask),otherwise_mask) # 1 1024 512
            otherwise_mask=torch.where(seg_result!=0,torch.ones_like(otherwise_mask),otherwise_mask)
            return otherwise_mask.unsqueeze(1)

        return mask.unsqueeze(1),choose_text#,class_id
   
    def seg_bg(self,seg_result):
        cloth=[1,2,3,4,5,6,21]
        mask=torch.ones_like(seg_result)
        
        for i in cloth:
            mask=torch.where(seg_result==i,torch.zeros_like(mask),mask) # 1 1024 512
        return mask.unsqueeze(1)

    def cal_avg(self,input,mask):
        #print('mask',mask.shape)
        #print('input',input.shape)
        x = input * mask
        sum = torch.sum(torch.sum(x, dim=2, keepdim=True), dim=3, keepdim=True) # [n,3,1,1]
        mask_sum = torch.sum(torch.sum(mask, dim=2, keepdim=True), dim=3, keepdim=True) # [n,1,1,1]
        mask_sum[mask_sum == 0] = 1
        avg = sum / mask_sum
        
       # print(avg)
        #print(t)
        return avg
    def crop_image(self,img,sample_size,part="upper"):
        cropped=torch.zeros(img.shape[0],sample_size,sample_size,3).cuda()
        
        #upper
        if part=="upper":
            i=257
            j=200
        #lower
        elif part=="lower":
            i=515
            j=187
        cropped=img[:,i:i+sample_size,j:j+sample_size]

        return cropped
    
    def image_color_loss_only(self,patch,x_hat,part="upper"):
        mask_fake=torch.ones_like(patch)
        mask_real=torch.ones_like(patch)
        img_tensor=patch
        gen_crop=self.crop_image(x_hat.permute(0,2,3,1),img_tensor.shape[-1],part=part) # B 64 64 3
        gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
        color_loss=self.calc_color_loss(img_tensor,gen_crop,mask_real,mask_fake)
        return color_loss

    def loss_image_texture(self,patch,x_hat,part="upper"):
       
        mask_fake=torch.ones_like(patch)
        mask_real=torch.ones_like(patch)
       
        #perceptual loss
        img_tensor=patch
        gen_crop=self.crop_image(x_hat.squeeze().permute(1,2,0),img_tensor.shape[-1],part=part) # 1 64 64 3
        gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
        
        percept_loss = 250*self.style_loss(img_tensor,gen_crop).mean()
       
        color_loss=self.calc_color_loss(img_tensor,gen_crop,mask_real,mask_fake)
       
        return color_loss,percept_loss
    
    def perceptual_loss_only(self,fake,real,part="upper"):
        #perceptual loss
        img_tensor=fake
        gen_crop=self.crop_image(real.permute(0,2,3,1),img_tensor.shape[-1],part=part) # B 64 64 3
        gen_crop=gen_crop.permute(0,3,1,2) # 1 3 64 64
        
        percept_loss = 250*self.style_loss(img_tensor,gen_crop).mean()
        #color_loss=0.
        if self.opts.texture_loss_type=='clip':
            return gen_crop
        else:
            return percept_loss


    def calc_color_loss(self,x,x_hat,mask_x,mask_x_hat):
        x_hat_RGB=(x_hat+1)/2.0
        x_RGB=(x+1)/2.0

        # from RGB to Lab by liuqk
        x_xyz = self.rgb2xyz(x_RGB)
        x_Lab = self.xyz2lab(x_xyz)
        x_hat_xyz = self.rgb2xyz(x_hat_RGB)
        x_hat_Lab = self.xyz2lab(x_hat_xyz)

        # cal average value
        x_Lab_avg = self.cal_avg(x_Lab, mask_x)
        x_hat_Lab_avg = self.cal_avg(x_hat_Lab, mask_x_hat)

        color_loss = self.color_l1_loss(x_Lab_avg, x_hat_Lab_avg)

        return color_loss

    def loss_skin_only(self,x,x_hat):
        seg_x, seg_x_hat= self.get_seg_only(x,x_hat)
        #if self.seg_model=='clip':
        skin_text=['skin']
        mask_x,_=self.seg_with_text(seg_x,skin_text)
        mask_x_hat, _=self.seg_with_text(seg_x_hat,skin_text)
        color_loss=self.calc_color_loss(x,x_hat,mask_x,mask_x_hat)
        if self.opts.background_lambda>0:
            mask_x_bg=self.seg_bg(seg_x)
            mask_x_hat_bg=self.seg_bg(seg_x_hat)
            mask=mask_x_bg*mask_x_hat_bg
            bg_loss=self.bg_mask_l2_loss(x*mask, x_hat*mask)
            return (color_loss,bg_loss)
        return color_loss
    
    def get_seg_only(self,x,x_hat):
        if self.seg_model=='clip':
            skin_text=['skin']
            '''
            CLASSES = ('background', 'top','outer','skirt','dress','pants','leggings','headwear',
                    'eyeglass','neckwear','belt','footwear','bag','hair','face','skin',
                    'ring','wrist wearing','socks','gloves','necklace','rompers','earrings','tie')
            '''
            with torch.no_grad():
                seg_result_x=self.model.simple_test(x,data_meta) # 1 1024 512
                seg_result_x_hat=self.model.simple_test(x_hat,data_meta) # 1 1024 512
            
        else:
            skin_text=['Torso']
            seg_result_x=self.model.get_seg(x.squeeze().permute(1,2,0).detach().cpu().numpy())
            seg_result_x_hat=self.model.get_seg(x_hat.squeeze().permute(1,2,0).detach().cpu().numpy())

        seg_result_x=torch.from_numpy(np.array(seg_result_x)).cuda()
        seg_result_x_hat=torch.from_numpy(np.array(seg_result_x_hat)).cuda()

        return seg_result_x,seg_result_x_hat
    
   