# Kelly McKernan
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Kelly McKernan' \
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
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_KellyMcKernan-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/short_niche_art_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_KellyMcKernan-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/long_niche_art_prompts.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 5

# Thomas Kinkade
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Thomas Kinkade' \
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
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_ThomasKinkade-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/short_niche_art_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_ThomasKinkade-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/long_niche_art_prompts.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 5

# Ajin Demi Human
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Ajin Demi Human' \
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
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_AjinDemiHuman-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/short_niche_art_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_AjinDemiHuman-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/long_niche_art_prompts.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 5

# Tyler Edlin
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Tyler Edlin' \
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
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_TylerEdlin-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/short_niche_art_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_TylerEdlin-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/long_niche_art_prompts.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 5
# Kilian Eng
python train_adversarial_gumbel.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Kilian Eng' \
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
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_KilianEng-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/short_niche_art_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='compvis-adversarial-gumbel-word_KilianEng-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --prompts_path 'data/long_niche_art_prompts.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 5
