import json
from torch.utils.data import Dataset
import numpy as np
import clip
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import os
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader


ABC_type={
	'A':['dress','romper','jumpsuit'],
    'B':['top','shirt'],
    'C':['pants','skirt','joggers'],
    'one':["long sleeve","short sleeve","sleeveless","tank","camisole","no"],
    'two':["round-neck","v-neck","collared","polo","no"],
}

class LatentsDataset(Dataset):

	def __init__(self, opts, status='train'):
	
		self.opts = opts
		self.status = status
		
		self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		if self.status == 'train':
			
			with open(self.opts.color_ref_img_train_path,"rb") as f:
				self.out_domain_color_img_path_list=pickle.load(f)	
			with open(self.opts.data_train,"rb") as f:
				self.data_list=pickle.load(f)

			with open(self.opts.img_latent_label_train,"rb") as f:
				self.latent_labels=pickle.load(f)
				
		else:
			
			with open(self.opts.color_ref_img_test_path,"rb") as f:
				self.out_domain_color_img_path_list=pickle.load(f)
			with open(self.opts.data_test,"rb") as f:
				self.data_list=pickle.load(f)
			with open(self.opts.img_latent_label_test,"rb") as f:
				self.latent_labels=pickle.load(f)

		if self.opts.test:
			with open(self.opts.test_data_list,"r") as f:
				self.test_data_list=json.load(f)
	
	def change_type_ABC(self):
		if np.random.random()<0.3:
			A_target=random.choice(ABC_type["A"])
			one_target=random.choice(ABC_type["one"])
			two_target=random.choice(ABC_type["two"])
			up_target=' '.join([one_target,two_target,A_target])
			lower_target=A_target
			all_target=' '.join([one_target,two_target,A_target])
			return (up_target.replace('no ',''),lower_target.replace('no ','')), [all_target.replace('no ','')]
		
		B_target=random.choice(ABC_type["B"])
		C_target=random.choice(ABC_type["C"])
		one_target=random.choice(ABC_type["one"])
		two_target=random.choice(ABC_type["two"])
		upper_target=' '.join([one_target,two_target,B_target])
		lower_target=C_target
		return '',(upper_target.replace('no ',''),lower_target.replace('no ',''))

	def crop_image(self,img,sample_size,part="upper"):
		cropped=torch.zeros(sample_size,sample_size,3)
		if part=="upper":
			i=257
			j=200
        #lower
		elif part=="lower":
			i=515
			j=187
		
		cropped=img[i:i+sample_size,j:j+sample_size]

		return cropped

	def manipulate_type(self, index):

		
		if self.opts.test:
			image_name=self.test_data_list[index]['img']
		else:
			image_name=self.data_list[index].split('.')[0]
		
		latent=torch.from_numpy(self.latent_labels[image_name]['latent'])
		
		ori_type=self.latent_labels[image_name]['cloth_type']
		
		# choose target type
		if self.opts.test:
			one_piece_parts,target_type=self.test_data_list[index]['text']
		else:
			one_piece_parts,target_type=self.change_type_ABC()
		
		# onepiece or two parts
		text_description=' and '.join(target_type)
		if len(target_type)==1:
			target_type=one_piece_parts
		
		type_text_emb_up=torch.cat([clip.tokenize(target_type[0])])[0]
		type_text_emb_low=torch.cat([clip.tokenize(target_type[1])])[0]
		
		return latent, type_text_emb_up, type_text_emb_low, text_description, target_type, ori_type

	def choose_self_texture(self,index,part="upper"):
		if self.opts.test:
			image_name=self.test_data_list[index]['img']
		else:
			image_name=self.data_list[index].split('.')[0]
		
		image_file=os.path.join(self.opts.real_imgs_dir,'aligned_{}'.format(self.status),'{}.png'.format(image_name))
		
		img = self.image_transform(Image.open(image_file)) 
		#print(img.shape)
		crop_gen=self.crop_image(img.permute(1,2,0),64,part=part)#64 64 3
		color_tensor=crop_gen.permute(2,0,1)
		return color_tensor

	def choose_ref_texture(self,index,part="upper", choose_imagename=None):
		
		if choose_imagename==None:
			choose_imagename=random.choice(self.out_domain_color_img_path_list)
		img_pil = Image.open(os.path.join(self.opts.texture_img_dir,choose_imagename))
		color_tensor = self.image_transform(img_pil)
		return color_tensor

	def manipulater_color(self, index):
		
		if self.opts.test:
			image_name=self.test_data_list[index]['img']
			choose_textures=self.test_data_list[index]['texture']

			if choose_textures[0]=='self':
				color_tensor1=self.choose_self_texture(index,part="upper")
			else:
				color_tensor1=self.choose_ref_texture(index,choose_imagename=choose_textures[0])

			if choose_textures[1]=='self':
				color_tensor2=self.choose_self_texture(index,part="lower")
			else:
				color_tensor2=self.choose_ref_texture(index,part="lower",choose_imagename=choose_textures[1])

		else:
			
			if random.random()<0.2:
				color_tensor1=self.choose_self_texture(index,part="upper")
			else:
				color_tensor1=self.choose_ref_texture(index)

			if random.random()<0.2:
				color_tensor2=self.choose_self_texture(index,part="lower")
			else:
				color_tensor2=self.choose_ref_texture(index,part="lower")
			
		return color_tensor1, color_tensor2
	
	def manipulater_type_and_texture(self, index): 
		
		latent, type_text_emb_up, type_text_emb_low, text_description, target_type, ori_type=self.manipulate_type(index)
		color_tensor_up, color_tensor_low=self.manipulater_color(index)
		selected_description = text_description
		
		if self.opts.no_medium_mapper:
			if ori_type[0]==ori_type[1]:
				color_tensor_low=color_tensor_up
		elif ("dress" in selected_description) or ("romper" in selected_description)  \
		    or ("overalls" in selected_description) or ("jumpsuit" in selected_description):
			color_tensor_low=color_tensor_up

		if self.opts.test:
			return latent, type_text_emb_up, type_text_emb_low,selected_description, color_tensor_up,color_tensor_low, target_type, self.test_data_list[index]["img"]

		
		return latent, type_text_emb_up, type_text_emb_low,selected_description, color_tensor_up,color_tensor_low, target_type, ori_type
	
	def __len__(self):
		if self.opts.test:
			return len(self.test_data_list)
		else:
			return len(self.data_list)

	def __getitem__(self, index):
		return self.manipulater_type_and_texture(index)

class FashiondataModule(pl.LightningDataModule):
    def __init__(self, opts, batch_size = 4):
        super().__init__()
        self.opts=opts
        self.batch_size = opts.batch_size
        self.train_data = LatentsDataset(opts)
        self.test_data=LatentsDataset(opts,status='test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,num_workers=8,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,num_workers=4,drop_last=True)
	
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.opts.test_batch_size,num_workers=1)