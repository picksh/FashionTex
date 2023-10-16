cd mapper
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
--exp_name="released_version" \
--description="released_version" \
--output_dir='../outputs' \
--test \
--test_data_list='' \
--test_img_dir='' \
--test_texture_dir='' \
--checkpoint_path='' \