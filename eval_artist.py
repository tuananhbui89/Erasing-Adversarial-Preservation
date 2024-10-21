import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os

# Data on Trung's server
data_path = dict()
prompt_name = 'long_niche_art_prompts'
SD_path = 'artist/SD-v1-4/{}/'.format(prompt_name)
data_path['ESD'] = 'artist/diffusers-{}-ESDx1-UNET/{}/'
data_path['UCE'] = 'artist/UCE/uce-erased-{}-towards_uncond-preserve_true-sd_1_4-method_tensor-info/{}/'
data_path['AE'] = 'evaluation_massive/compvis-adversarial-gumbel-word_{}-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000/{}/'
num_samples = 5

art_list = ['KellyMcKernan', 'KilianEng', 'TylerEdlin', 'ThomasKinkade', 'AjinDemiHuman']

df =  pd.read_csv('data/{}.csv'.format(prompt_name))

# ---------------------
import lpips
import math
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

for method in data_path.keys():
    # create new data frame, each for a method
    # each row: to-be-erased-artist, prompt, num, lpips score
    # total row: len(art_list) * len(df) * num_samples

    result = pd.DataFrame(columns=['erased_artist', 'prompt_artist', 'prompt', 'num', 'lpips_score'])

    for artist in art_list:

        for _, row in df.iterrows():

            for num in range(num_samples):
                  
                path = data_path[method].format(artist, prompt_name)
                prompt = row['prompt']
                case_number = row['case_number']
                art = row['artist'].replace(' ', '').replace(':', '').lower()

                tar_file = path + f'{case_number}_{num}.png'
                org_file = SD_path + f'{case_number}_{num}.png'

                if not os.path.exists(tar_file) or not os.path.exists(org_file):
                    if not os.path.exists(tar_file):
                        print(f'File not found: {tar_file}')
                    if not os.path.exists(org_file):
                        print(f'File not found: {org_file}')
                    lpips_score = np.nan
                else:
                    tar_img = lpips.im2tensor(lpips.load_image(tar_file))
                    org_img = lpips.im2tensor(lpips.load_image(org_file))
                    lpips_score = loss_fn_alex(tar_img, org_img).item()

                    if math.isnan(lpips_score):
                        print(f'nan: {tar_file} or {org_file}')
                        break

                print(f'{method} {artist} {prompt} {num} {lpips_score}')
                result = result._append({'erased_artist': artist,'prompt_artist': art, 'prompt': prompt, 'num': num, 'lpips_score': lpips_score}, ignore_index=True)
    assert len(result) == len(art_list) * len(df) * num_samples
    # write to csv
    result.to_csv(f'evaluation_folder/artist_{method}_{prompt_name}_lpips.csv', index=False)


# --------------------- CLIP score ---------------------

import pandas as pd
import os
from PIL import Image
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision.transforms.functional import pil_to_tensor, resize
import torch

metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric.to(device)

for method in data_path.keys():
    # create new data frame, each for a method
    # each row: to-be-erased-artist, prompt, num, score1: clip score with prompt, score2: clip score with artist, score3: clip score with prompt without artist
    # total row: len(art_list) * len(df) * num_samples

    result = pd.DataFrame(columns=['erased_artist', 'prompt_artist', 'prompt', 'score1', 'score2', 'score3'])

    for artist in art_list:
        
        path = data_path[method].format(artist, prompt_name)

        for _, row in df.iterrows():

            for num in range(num_samples):

                prompt = row['prompt']
                case_number = row['case_number']
                art = row['artist'].replace(' ', '').replace(':', '').lower()

                tar_file = path + f'{case_number}_{num}.png'

                if not os.path.exists(tar_file):
                    print(f'File not found: {tar_file}')
                    score1 = np.nan
                    score2 = np.nan
                    score3 = np.nan
                
                else:
                    image = pil_to_tensor(Image.open(tar_file))
                    image = resize(image, 224, antialias=True)
                    image = image.unsqueeze(0)
                    image = image.to(device)
                    with torch.no_grad():
                        score1 = metric(image, prompt).item()
                        # reset memory to avoid OOM
                        # torch.cuda.empty_cache()
                        score2 = metric(image, row['artist']).item()
                        # torch.cuda.empty_cache()
                        score3 = metric(image, prompt.replace(row['artist'], '')).item()
                print(f'{method} {artist} {prompt} {num} {score1} {score2} {score3}')
                result = result._append({'erased_artist': artist, 'prompt_artist': art, 'prompt': prompt, 'score1': score1, 'score2': score2, 'score3': score3}, ignore_index=True)
    
    assert len(result) == len(art_list) * len(df) * num_samples
    result.to_csv(f'evaluation_folder/artist_{method}_{prompt_name}_clip.csv', index=False)

# run clip one more time for SD-v1-4
# create new data frame for SD
# each row: artist, prompt, num, score1: clip score with prompt, score2: clip score with artist, score3: clip score with prompt without artist
# total row: 1 * len(df) * num_samples

result = pd.DataFrame(columns=['erased_artist', 'prompt_artist', 'prompt', 'score1', 'score2', 'score3'])
for _, row in df.iterrows():
    prompt = row['prompt']
    case_number = row['case_number']
    art = row['artist'].replace(' ', '').replace(':', '').lower()

    for num in range(num_samples):
        tar_file = SD_path + f'{case_number}_{num}.png'

        if not os.path.exists(tar_file):
            print(f'File not found: {tar_file}')
            score1 = np.nan
            score2 = np.nan
            score3 = np.nan
        else:
            image = pil_to_tensor(Image.open(tar_file))
            image = resize(image, 224, antialias=True)
            image = image.unsqueeze(0)
            image = image.to(device)
            with torch.no_grad():
                score1 = metric(image, prompt).item()
                # reset memory to avoid OOM
                torch.cuda.empty_cache()
                score2 = metric(image, row['artist']).item()
                torch.cuda.empty_cache()
                score3 = metric(image, prompt.replace(row['artist'], '')).item()
        print(f'SD {art} {prompt} {num} {score1} {score2} {score3}')
        result = result._append({'erased_artist': 'None', 'prompt_artist': art, 'prompt': prompt, 'score1': score1, 'score2': score2, 'score3': score3}, ignore_index=True)
assert len(result) == len(df) * num_samples
result.to_csv(f'evaluation_folder/artist_SD_{prompt_name}_clip.csv', index=False)

