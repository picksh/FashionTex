import torch
import numpy as np
from PIL import Image
import sys
import os
import torchvision.transforms as transforms

from denseclip import DenseCLIP
from configs.denseclip_fpn_vit_b_640x640_80k import CONF,data_meta

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


img_trans=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

cpk_file='/data_sobig/anran/DenseCLIP/segmentation/work_dirs/denseclip_fpn_vit-b_640x640_80k/iter_80000.pth'

model=DenseCLIP(**CONF)
model.load_state_dict(torch.load(cpk_file)['state_dict'])
model.cuda()
model.eval()
img_path='/data_sobig/anran/deepfashionmm/edit_w_image_test'
save_dir='/data_sobig/anran/deepfashionmm/edit_w_seg'
image_list=os.listdir(img_path)
image_list=[x for x in image_list if ".png" in x and 'alpha' not in x]

paletee=get_palette(24)
for img_name in image_list:
    print(img_name)
    img=Image.open(os.path.join(img_path,img_name))
    img=img_trans(img)
    img=img.unsqueeze(0).cuda()
    #img=torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2).float().cuda()
    with torch.no_grad():
        seg_result_x=model.simple_test(img,data_meta)
    #print(seg_result_x[0].max())
    #print(seg_result_x[0].min())
    #print(max(seg_result_x))
    #print(min(seg_result_x))
    #print(np.array(seg_result_x).shape)
    np.save(os.path.join(save_dir,'{}.npy'.format(img_name.split('.')[0])),seg_result_x[0])
    s=Image.fromarray(np.asarray(seg_result_x[0], dtype=np.uint8))
    s.putpalette(paletee)
    s.save(os.path.join(save_dir,img_name))

