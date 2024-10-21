# Adversarial-Erasing

Code for the paper *"Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation"*.

## Installation Guide

```bash
cd Adversarial-Erasing
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4-full-ema.ckpt
mkdir models/erase
mv sd-v1-4-full-ema.ckpt models/erase/
wget https://huggingface.co/CompVis/stable-diffusion-v1-4/blob/main/unet/config.json
mv config.json models/erase/
```

Requirements:

```bash
pip install omegaconf
pip install pytorch-lightning==1.6.5
pip install taming-transformers-rom1504
pip install kornia==0.5.11
pip install git+https://github.com/openai/CLIP.git
pip install diffusers==0.21.4
pip install -U transformers
pip install --upgrade nudenet
pip install lpips
```

## Usage

We provide training and evaluation scripts for the experiments in the paper in the following bash files. 

To produce the results in Table 1 of the paper (i.e., Erasing object-related concepts), run the following command:

```bash
bash run_imagenette.sh
```

To produce the results in Table 2 of the paper (i.e., Erasing nudity concept), run the following command:

```bash
bash run_nudity.sh
```

To produce the results in Table 3 of the paper (i.e., Erasing Artistic Concepts), run the following command:

```bash
bash run_artist.sh
```

To produce Figure 1 and Figure 2 of the paper (i.e., Analysis of the impact of erasing the target concept)

```bash
bash run_abl_preserve.sh
```

The list of prompts used in the paper can be found in the `data` folder, including:

- `english_3000.csv`: List of 3000 English words
- `imagenette.csv`: List of imagenette classes, 500 images per class
- `unsafe-prompts4703.csv`: List of unsafe prompts I2P, 4703 prompts
- `long_nich_art_prompts.csv`: List to generate artistic from five artists
- `similarity-nudity_200.csv` to `similarity-nudity-4_200.csv`: List to generate specific objects to study the impact of erasing nudity and garbage truck concepts

We provide implementation of our method and baselines:

- `train_adversarial_gumbel.py`: Implementation of our method
- `train_esd.py`: Implementation of ESD 
- `train_uce.py`: Implementation of UCE
- `train-esd-preserve.py`: Implementation of ESD with preservation to study the impact of erasing nudity and garbage truck concepts

To set concepts to erase, modify the `utils_exp.py` file and change the argument `--prompt` in the bash files.

## References

This repository is based on the repository [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing)