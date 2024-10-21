# Generate images to investigate impact of earsing concept "nudity" or "garbage truck" on generating other concepts
# First, obtain fine-tuned model with the concept "nudity" or "garbage truck" erased
# Three models: original, "compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none", "compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated"
# One additional model: "compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve"

# Original model
python generate_images_ldm.py --models_path='' --model_name="SD-v1-4" --config_path 'configs/stable-diffusion/v1-inference.yaml' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='' --model_name="SD-v1-4" --config_path 'configs/stable-diffusion/v1-inference.yaml' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='' --model_name="SD-v1-4" --config_path 'configs/stable-diffusion/v1-inference.yaml' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='' --model_name="SD-v1-4" --config_path 'configs/stable-diffusion/v1-inference.yaml' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1

# MODEL "compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated"
python train-esd.py --seperator ',' --train_method 'noxattn' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --diffusers_config_path '../Better_Erasing/models/erase/config.json' --prompt 'nudity' --config_path 'configs/stable-diffusion/v1-inference.yaml' --info 'separated'

python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1


# MODEL "compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve"
python train-esd-preserve.py --seperator ',' --train_method 'noxattn' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --diffusers_config_path '../Better_Erasing/models/erase/config.json' --prompt 'nudity_with_person' --config_path 'configs/stable-diffusion/v1-inference.yaml' --info 'esd-preserve'

python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1

# MODEL "compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_500_EN3K"

python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_500_EN3K" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_500_EN3K" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_500_EN3K" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_500_EN3K" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1


# MODEL "compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none"
python train-esd.py --seperator ',' --train_method 'xattn' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --diffusers_config_path '../Better_Erasing/models/erase/config.json' --prompt 'garbage_truck' --config_path 'configs/stable-diffusion/v1-inference.yaml' --info 'none'

python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1 --from_case 370
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_none" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1


# MODEL "compvis-word_garbage_truck_with_lexus-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve"

python train-esd-preserve.py --seperator ',' --prompt 'garbage_truck_with_lexus' --train_method 'xattn' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --diffusers_config_path '../Better_Erasing/models/erase/config.json' --config_path 'configs/stable-diffusion/v1-inference.yaml' --info 'esd-preserve'

python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_lexus-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_lexus-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_lexus-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_lexus-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1

# MODEL "compvis-word_garbage_truck_with_road-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve"

python train-esd-preserve.py --seperator ',' --prompt 'garbage_truck_with_road' --train_method 'xattn' --ckpt_path '../Better_Erasing/models/erase/sd-v1-4-full-ema.ckpt' --diffusers_config_path '../Better_Erasing/models/erase/config.json' --config_path 'configs/stable-diffusion/v1-inference.yaml' --info 'esd-preserve'

python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_road-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_road-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_road-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-word_garbage_truck_with_road-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1


# MODEL "compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_100_test_2"

python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_100_test_2" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_100_test_2" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-2_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_100_test_2" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-3_200.csv'  --save_path 'evaluation_massive' --num_samples 1
python generate_images_ldm.py --models_path='models' --model_name="compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_100_test_2" --config_path 'configs/stable-diffusion/v1-inference.yaml' --prompts_path 'data/similarity-nudity-4_200.csv'  --save_path 'evaluation_massive' --num_samples 1

# MODEL "compvis-adversarial-gumbel-word_garbage_truck-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000_test_2"

python eval_simmilarity_clip.py

python gen_embedding_matrix.py --time_step 20 --task 'compare_two_models' --concept 'nudity' --vocab 'EN3K'
python gen_embedding_matrix.py --time_step 20 --task 'compare_two_models' --concept 'garbage_truck' --vocab 'EN3K'

python gen_embedding_matrix.py --time_step 20 --task 'compare_two_models' --concept 'nudity' --vocab 'CLIP'
python gen_embedding_matrix.py --time_step 20 --task 'compare_two_models' --concept 'garbage_truck' --vocab 'CLIP'



