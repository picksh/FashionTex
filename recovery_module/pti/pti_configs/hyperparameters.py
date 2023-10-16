## Architechture
lpips_type = 'alex'
first_inv_type = 'w+'
optim_type = 'adam'

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 100
pt_lpips_lambda = 1

## Steps
LPIPS_value_threshold = 0.005#0.04
max_pti_steps = 800#450#350
first_inv_steps = 450
max_images_to_invert = 30000

## Optimization
pti_learning_rate = 5e-4
first_inv_lr = 8e-3
train_batch_size = 1
use_last_w_pivots = False
