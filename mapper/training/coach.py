import os
import torch
import torchvision
from torch import nn
import json
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from criteria.human_parse import seg_loss
import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.clip_mapper import CLIPMapper
from mapper.training.ranger import Ranger
import numpy as np
from PIL import Image
import numpy as np
import pytorch_lightning as pl

import warnings
warnings.filterwarnings('ignore')

class Coach(pl.LightningModule):
	def __init__(self, opts):
		super().__init__()
		self.save_hyperparameters()
		self.opts = opts
		# Initialize network
		self.net = CLIPMapper(self.opts).to(self.device)
		# Initialize loss
		self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		self.clip_loss = clip_loss.CLIPLoss(opts)
		self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
		self.seg_loss=seg_loss.SegLoss(opts,seg_model=opts.seg_model)

		# Initialize logger
		log_dir = os.path.join(opts.output_dir,opts.exp_name,'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir

		if not self.opts.test:
			opts_dict = vars(opts)
			with open(os.path.join(opts.output_dir, opts.exp_name,'opt.json'), 'w') as f:
				json.dump(opts_dict, f, indent=4, sort_keys=True)
		
		if opts.optim_name=='ranger':
			self.automatic_optimization = False

	
	def forward(self,w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low):
		
		w_hat=w+0.1*self.net.mapper(w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low)
		x_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)

		return x_hat, w_hat

	def _shared_eval(self, batch, batch_idx, prefix):
		
		w,  type_text_emb_up, type_text_emb_low, selected_description, color_tensor_up,color_tensor_low, target_type, image_label= batch
		with torch.no_grad():
			x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
		x_hat, w_hat=self(w, type_text_emb_up, color_tensor_up, color_tensor_low, type_text_emb_low)

		if self.opts.test:
			return x,x_hat,color_tensor_up,color_tensor_low,selected_description, image_label, w_hat

		loss, loss_dict = self.calc_loss(w, x, w_hat, x_hat, color_tensor_up,color_tensor_low, target_type, image_label, log_seg=False, prefix=prefix)
		self.log_dict(loss_dict,on_step=(prefix=="train"), on_epoch=True, prog_bar=True)
		
		return loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low, selected_description

	def training_step(self,batch,batch_idx):
		if self.opts.optim_name=='ranger':
			optimizer = self.optimizers()
			optimizer.zero_grad()
		
		loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low, selected_description=self._shared_eval(batch,batch_idx,prefix="train")
		
		if self.opts.optim_name=='ranger':
			self.manual_backward(loss)
			optimizer.step()
		
		if self.global_step % self.opts.image_interval == 0 or (
				self.global_step < 1000 and self.global_step % 1000 == 0):
			self.log_loss(loss_dict)
			self.parse_and_log_images(x, x_hat, color_tensor_up,color_tensor_low, title='images_train', selected_description=selected_description)

		return loss
		
	def validation_step(self,batch,batch_idx):
		
		loss, loss_dict, x, x_hat, color_tensor_up,color_tensor_low,selected_description=self._shared_eval(batch,batch_idx,prefix="test")
		
		if batch_idx%100==0:
			self.parse_and_log_images(x, x_hat, color_tensor_up,color_tensor_low ,title='images_val', selected_description=selected_description, index=batch_idx)
			self.log_loss(loss_dict)
		
		return loss

	def test_step(self, batch, batch_idx):
		x,x_hat,color_tensor_up,color_tensor_low,selected_description, image_name, w_hat=self._shared_eval(batch, batch_idx, prefix='test')
		#x,x_hat,color_tensor,color_tensor2,selected_description, image_name, w_hat=self._shared_eval(batch, batch_idx, prefix='test')
		# Only save final results
		self.log_image(x_hat, image_name, selected_description, title='images_test/img')
		# Use self.parse_and_log_images to save original img, output, and ref textrues.
		
		# Save w_hat for recovery module
		self.log_w(w_hat, image_name, selected_description, title='images_test/w')

	def configure_optimizers(self):	
		
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(self.net.mapper.parameters(), lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(list(self.net.mapper.parameters()), lr=self.opts.learning_rate)
		return optimizer

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                   optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
		optimizer.step(closure=optimizer_closure)

	def calc_loss(self, w, x, w_hat, x_hat, color_tensor_up, color_tensor_low, target_type, image_label, log_seg=False,prefix='',real_image=None,des=None):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement = self.id_loss(x_hat, x)
			loss_dict[f"{prefix}_loss_id"] = float(loss_id* self.opts.id_lambda * self.opts.attribute_preservation_lambda)
			loss = loss_id * self.opts.id_lambda * self.opts.attribute_preservation_lambda
		
		if self.opts.latent_l2_lambda > 0:
			loss_l2_latent = self.latent_l2_loss(w_hat, w)
			loss_dict[f"{prefix}_loss_l2_latent"] = float(loss_l2_latent* self.opts.latent_l2_lambda * self.opts.attribute_preservation_lambda)
			loss += loss_l2_latent * self.opts.latent_l2_lambda * self.opts.attribute_preservation_lambda
		
		if self.opts.skin_lambda > 0:
			loss_skin=self.seg_loss.loss_skin_only(x_hat, x)
			
			if self.opts.background_lambda>0:
				loss_bg=loss_skin[1]
				loss_skin=loss_skin[0]
				
				loss_dict[f"{prefix}_loss_bg"]=float(loss_bg*self.opts.background_lambda*self.opts.attribute_preservation_lambda)
				loss+=loss_bg*self.opts.background_lambda*self.opts.attribute_preservation_lambda

			loss_dict[f"{prefix}_loss_skin"]=float(loss_skin*self.opts.skin_lambda*self.opts.attribute_preservation_lambda)
			loss+=loss_skin*self.opts.skin_lambda*self.opts.attribute_preservation_lambda
		
		loss_perceptual1=self.seg_loss.perceptual_loss_only(color_tensor_up,x_hat,part="upper")
		loss_perceptual2=self.seg_loss.perceptual_loss_only(color_tensor_low,x_hat,part="lower")
		loss_perceptual=(loss_perceptual1+loss_perceptual2)/2

		if self.opts.image_color_lambda>0:
			loss_color1=self.seg_loss.image_color_loss_only(color_tensor_up,x_hat,part="upper")
			loss_color2=self.seg_loss.image_color_loss_only(color_tensor_low,x_hat,part="lower")
			loss_color=(loss_color1+loss_color2)/2
			loss_dict[f"{prefix}_loss_image_color"]=float(loss_color*self.opts.image_color_lambda)
			loss+= loss_color*self.opts.image_color_lambda
			
		loss_dict[f"{prefix}_loss_perceptual"]=float(loss_perceptual*self.opts.perceptual_lambda*self.opts.image_manipulation_lambda)
		loss+= loss_perceptual*self.opts.perceptual_lambda*self.opts.image_manipulation_lambda
			
		loss_type=self.clip_loss.new_clip_loss(x,image_label,x_hat,target_type)
		loss_dict[f"{prefix}_loss_type"] = float(loss_type* self.opts.text_manipulation_lambda)
		loss += loss_type * self.opts.text_manipulation_lambda
		
		loss_dict[f"{prefix}_loss"] = float(loss)
		
		return loss, loss_dict

	def parse_and_log_images(self, img, img_hat, color_tensor ,color_tensor2, title, selected_description, index=None, real_image=None):
		x=img[0].unsqueeze(0).detach().cpu()
		x_hat=img_hat[0].unsqueeze(0).detach().cpu()
		img_tensor=color_tensor[0].unsqueeze(0).detach().cpu()
		img_tensor2=color_tensor2[0].unsqueeze(0).detach().cpu()
		selected_description=selected_description[0]
		if self.opts.test:
			index=index[0]
		
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{selected_description}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{str(index).zfill(5)}-{selected_description}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		color_tensor_pad=(int((x.shape[3]-img_tensor.shape[3])/2),int((x.shape[3]-img_tensor.shape[3])/2),int((x.shape[2]-img_tensor.shape[2])/2),int((x.shape[2]-img_tensor.shape[2])/2))
		torchvision.utils.save_image(torch.cat([x, x_hat, F.pad(img_tensor,pad=color_tensor_pad),F.pad(img_tensor2,pad=color_tensor_pad)]), path,
								     normalize=True, scale_each=True, range=(-1, 1), nrow=4)
	# TODO: May Need modify according to different output
	def log_image(self,x_hat,image_name,des, title):
		image_name=image_name[0].split('.')[0]
		des=des[0]
		path=os.path.join(self.log_dir,title, f'{image_name}+{des}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torchvision.utils.save_image(torch.cat([x_hat.detach().cpu()]),path,normalize=True, scale_each=True, range=(-1, 1))
	# TODO: May Need modify according to different output
	def log_w(self, w, image_name, des, title):
		image_name=image_name[0].split('.')[0]
		des=des[0]
		path=os.path.join(self.log_dir,title, f'{image_name}+{des}.npy')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		np.save(path, w.cpu().numpy())

		
	def log_loss(self, loss_dict):
		with open(os.path.join(self.log_dir, 'timestamp.txt'), 'a') as f:
			f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))
