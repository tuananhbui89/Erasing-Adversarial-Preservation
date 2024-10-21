import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os


prompt_list = ['similarity-nudity_200', 'similarity-nudity-2_200', 'similarity-nudity-3_200', 'similarity-nudity-4_200']

for prompt_name in prompt_list:
    data_path = dict()

    data_path['SD-v1-4'] = 'evaluation_massive/SD-v1-4/ldm-{}/'.format(prompt_name)
    data_path['ESD'] = 'evaluation_massive/compvis-word_nudity-method_noxattn-sg_3-ng_1.0-iter_1000-lr_1e-05-info_separated/ldm-{}/'.format(prompt_name)
    data_path['ESD-preserve-person'] = 'evaluation_massive/compvis-word_nudity_with_person-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_esd-preserve/ldm-{}/'.format(prompt_name)    
    data_path['AE'] = 'evaluation_massive/compvis-adversarial-gumbel-word_nudity-method_noxattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_50_update_-1_timestep_0_multi_2_kclosest_200_EN3K/ldm-{}/'.format(prompt_name)

    if prompt_name == 'similarity-nudity_200':
        concepts = ['women', 'men', 'person', 'hat', 'apple', 'bamboo']
    elif prompt_name == 'similarity-nudity-2_200':
        concepts = ['notebooks', 'australians', 'president', 'boat', 'lexus', 'money', 'banana']
    elif prompt_name == 'similarity-nudity-3_200':
        concepts = ['naked', 'garbage truck', 'a photo', ' ']
    elif prompt_name == 'similarity-nudity-4_200':
        concepts = ['road', 'car', 'bus']
    else: 
        raise ValueError('Prompt name not recognized')

    df =  pd.read_csv('data/{}.csv'.format(prompt_name))

    num_samples = 1

    # --------------------- CLIP score ---------------------

    import pandas as pd
    import os
    from PIL import Image
    from torchmetrics.multimodal.clip_score import CLIPScore
    from torchvision.transforms.functional import pil_to_tensor, resize
    import torch

    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
    metric.to('cuda')

    os.makedirs('evaluation_folder/clip', exist_ok=True)

    for method in data_path.keys():
        # create new data frame, each for a method

        result = pd.DataFrame(columns=['prompt', 'score', 'case_number', 'file_path'], dtype=object)


        path = data_path[method]

        for ir, row in df.iterrows():

            for num in range(num_samples):

                prompt = row['prompt']
                case_number = row['case_number']

                file_path = path + f'{case_number}_{num}.png'

                if not os.path.exists(file_path):
                    print(f'File not found: {file_path}')
                    score = np.nan
                
                else:
                    image = pil_to_tensor(Image.open(file_path))
                    image = resize(image, 224, antialias=True)
                    image = image.unsqueeze(0)
                    image = image.to('cuda')
                    with torch.no_grad():
                        score = metric(image, prompt).item()

                print(f'{method} {prompt} {ir} / {len(df)} {num} {score}')
                try:
                    result = result._append({'prompt': prompt, 'score': score, 'case_number': case_number, 'file_path': file_path}, ignore_index=True)
                except:
                    result = result.append({'prompt': prompt, 'score': score, 'case_number': case_number, 'file_path': file_path}, ignore_index=True)

        
        result.to_csv(f'evaluation_folder/clip/similarity_{method}_{prompt_name}_clip.csv', index=False)

