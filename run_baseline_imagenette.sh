python train-uce.py --prompt 'imagenette_v1_wo' --technique 'tensor' --concept_type 'object' --base '1.4' --add_prompts False --info 'none'

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v1_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v1_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/uce-erased-imagenette_v1_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/uce-erased-imagenette_v1_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none-imagenette.csv'


python train-uce.py --prompt 'imagenette_v2_wo' --technique 'tensor' --concept_type 'object' --base '1.4' --add_prompts False --info 'none'

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v2_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v2_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/uce-erased-imagenette_v2_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/uce-erased-imagenette_v2_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none-imagenette.csv'


python train-uce.py --prompt 'imagenette_v3_wo' --technique 'tensor' --concept_type 'object' --base '1.4' --add_prompts False --info 'none'

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v3_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v3_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/uce-erased-imagenette_v3_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/uce-erased-imagenette_v3_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none-imagenette.csv'


python train-uce.py --prompt 'imagenette_v4_wo' --technique 'tensor' --concept_type 'object' --base '1.4' --add_prompts False --info 'none'

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v4_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/small_imagenet_prompts.csv' \
    --save_path 'evaluation_folder' \
    --num_samples 1

python eval-scripts/generate-images.py \
    --models_path=models \
    --model_name='uce-erased-imagenette_v4_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none' \
    --prompts_path 'data/imagenette.csv' \
    --save_path 'evaluation_massive' \
    --num_samples 1 \
    --from_case 0

python eval-scripts/imageclassify.py \
    --topk=10 \
    --folder_path='evaluation_massive/uce-erased-imagenette_v4_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none/imagenette/' \
    --prompts_path='data/imagenette.csv' \
    --save_path='evaluation_folder/uce-erased-imagenette_v4_wo-towards_uncond-preserve_true-sd_1_4-method_tensor-info_none-imagenette.csv'
