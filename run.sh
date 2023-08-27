cd mapper
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py \
--exp_name="released_version" \
--description="released_version" \
--text_manipulation_lambda=80 \
--id_lambda=0.1 \
--latent_l2_lambda=0.8 \
--background_lambda=100 \
--skin_lambda=0.1 \
--perceptual_lambda=8 \
--image_color_lambda=1.0 \
--batch_size=8 \