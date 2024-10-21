# imagenette_v1_wo, pgd_num_steps=2
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path '../Better_Erasing/models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --pgd_num_steps 2 \
    --info 'gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_2' \

# imagenette_v1_wo, pgd_num_steps=4
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path '../Better_Erasing/models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --pgd_num_steps 4 \
    --info 'gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_4' \

# imagenette_v1_wo, pgd_num_steps=8
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'imagenette_v1_wo' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path '../Better_Erasing/models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --pgd_num_steps 8 \
    --info 'gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_8' \

# imagenette_v1_wo, pgd_num_steps=2
python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_2' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_2' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_2/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_2-ldm-imagenette.csv' \


# imagenette_v1_wo, pgd_num_steps=4
python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_4' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_4' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_4/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_4-ldm-imagenette.csv' \


# imagenette_v1_wo, pgd_num_steps=8
python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_8' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python generate_images_ldm.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_8' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_8/ldm-imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/compvis-adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_v2_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_pgd_8-ldm-imagenette.csv' \

