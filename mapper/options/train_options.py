import os
from argparse import ArgumentParser
WEIGHT_DIR="../pretrained "
DATA_DIR="../data"
Texture_ref_DIR="../data_texture"

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_name', type=str, help='exp name')
		self.parser.add_argument('--no_coarse_mapper', default=True)
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument("--output_dir",type=str,default="outputs")
		
		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')
		self.parser.add_argument('--learning_rate', default=0.0005, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
		self.parser.add_argument('--text_manipulation_lambda', default=2.0, type=float, help='Text manipulation loss multiplier factor')
		self.parser.add_argument('--image_manipulation_lambda', default=1.0, type=float, help='Image manipulation loss multiplier factor')
		self.parser.add_argument('--attribute_preservation_lambda', default=1.0, type=float, help='Attribute preservation loss multiplier factor')
		self.parser.add_argument('--image_color_lambda', default=0.02, type=float, help='Image-based color manipulation loss multiplier factor')
		self.parser.add_argument('--id_lambda', default=0.3, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--maintain_color_lambda', default=0.02, type=float, help='Color retention loss multiplier factor')
		self.parser.add_argument('--background_lambda', default=1.0, type=float, help='Background loss multiplier factor')
		self.parser.add_argument('--latent_l2_lambda', default=0.8, type=float, help='Latent L2 loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default=os.path.join(WEIGHT_DIR,'stylegan_human_v2_1024.pth'), type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--ir_se50_weights', default=os.path.join(WEIGHT_DIR,'model_ir_se50.pth'), type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint')

		self.parser.add_argument('--max_steps', default=2000000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=500, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=2000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=2000, type=int, help='Model checkpoint interval')

		self.parser.add_argument("--img_latent_label_train",type=str,default=os.path.join(DATA_DIR,"data_split/emb_label_train_final.pkl"))
		self.parser.add_argument("--img_latent_label_test",type=str, default=os.path.join(DATA_DIR,"data_split/emb_label_test_final.pkl"))
		self.parser.add_argument("--data_train",type=str, default=os.path.join(DATA_DIR,"data_split/deepfashionmm_train.pkl"))
		self.parser.add_argument("--data_test",type=str, default=os.path.join(DATA_DIR,"data_split/deepfashionmm_test.pkl"))
		self.parser.add_argument("--real_imgs_dir",type=str,default=os.path.join(DATA_DIR,'data_split/aligned'))
		self.parser.add_argument('--color_ref_img_train_path', default=os.path.join(Texture_ref_DIR,"texture_img_train.pkl"), type=str)
		self.parser.add_argument('--color_ref_img_test_path', default=os.path.join(Texture_ref_DIR,"texture_img_test.pkl"), type=str)

		self.parser.add_argument("--seg_model",default='clip',type=str,help='choose from "clip" and "SCHP"')
		self.parser.add_argument("--seg_model_path",default=os.path.join(WEIGHT_DIR,'iter_80000.pth'))
		self.parser.add_argument("--description",type=str)
		self.parser.add_argument("--log_train_seg",action='store_true')
		self.parser.add_argument("--skin_lambda",default=1.0,type=float)
		self.parser.add_argument("--fast_dev_run",action="store_true")
		self.parser.add_argument("--resume_training",action="store_true")

		#For texture edit
		self.parser.add_argument("--texture_img_dir", type=str, default=os.path.join(DATA_DIR, "texture_64"))
		self.parser.add_argument("--perceptual_lambda",default=1.0,type=float)
		self.parser.add_argument("--texture_loss_type",type=str,default='lpip',help='lpip,clip')

		# For type Edit
		self.parser.add_argument("--cliploss_type",type=str,default="new_cliploss",help=['clip,seg_clip,new_cliploss,classification_cliploss'])

		# For test 
		self.parser.add_argument("--test",action="store_true")
		self.parser.add_argument("--test_data_list", type=str)
		self.parser.add_argument("--test_img_dir", type=str)
		self.parser.add_argument("--test_texture_dir", type=str)

	def parse(self):
		opts = self.parser.parse_args()
		return opts