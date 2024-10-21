# ablation study on the number of centers
# 100 centers
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

# 200 centers
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 200 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000' \

# 50 centers
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 50 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_1000' \

# 20 centers
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 20 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_20_update_-1_timestep_0_multi_2_kclosest_1000' \


# 200 centers
python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_1_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000-ldm-imagenette.csv' \

# 50 centers
python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_1000/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_1000-ldm-imagenette.csv'

# 20 centers
python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_20_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_20_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_20_update_-1_timestep_0_multi_2_kclosest_1000/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_20_update_-1_timestep_0_multi_2_kclosest_1000-ldm-imagenette.csv'