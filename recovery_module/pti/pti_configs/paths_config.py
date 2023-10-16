import os

## Pretrained models paths
e4e = './pti/e4e_w+.pt'
stylegan2_ada_shhq = '' # .pkl format
seg_model_path='' # same with the seg model with mapper


## Dirs for output files
out_dir='output'

checkpoints_dir = os.path.join(out_dir,'checkpoints') # can be ignored
embedding_base_dir = os.path.join(out_dir,'embeddings') # can be ignored
ori_embedding_base_dir =os.path.join(out_dir,'embeddings') # can be ignored
ir_se50 =  '' 

experiments_output_dir = os.path.join(out_dir,'results') # output dir # CHANGE HERE

os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(experiments_output_dir, exist_ok=True)

## Input info
### Input dir, where the images reside

# CHANGE HERE: replace with the output of mapper
input_data_path = '' # real image
edit_w_dir='' # w
edit_file_dir='' #img

### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator

input_data_id = 'person_id'  # can be ignored
## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_Plus'
multi_id_model_type = 'multi_id'
