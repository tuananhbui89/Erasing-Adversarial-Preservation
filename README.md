<div align="center">

## ⚡️Erasing-Adversarial-Preservation⚡️

Code for the paper *"Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation"* (accepted at NeurIPS 2024).

[[📄 Paper]](https://arxiv.org/abs/2410.15618) [[🌟 Project Page]](https://tuananhbui89.github.io/projects/adversarial-preservation/) [[🎨 Poster]](https://www.dropbox.com/scl/fi/tsb6036mb5mme7br73kr4/NeurIPS-2024-AP.pdf?rlkey=c4r1ecgxnm6xd3wy00vaprq62&st=15u48nos&dl=0) [[📊 Slides]](https://www.dropbox.com/scl/fi/jmqtqp939jfr7p3xrel8x/2024-AP-compact.pdf?rlkey=bzkhlhmg63efijlzo51mpnwzf&st=69no55kd&dl=0)

Contact: tuananh.bui@monash.edu

**Our other papers on Concept Erasing/Unlearning:**

> [**Fantastic Targets for Concept Erasure in Diffusion Models and Where to Find Them**](https://www.dropbox.com/scl/fi/pf2190qpfpiuo05mhcqmi/Adaptive-Guide-Erasure.pdf?rlkey=63s7ruwqxhrdsc4i603gjmsri&st=y79mr0ej&dl=0),
> Tuan-Anh Bui, Trang Vu, Long Vuong, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung
> *Under Review ([Dropbox](https://www.dropbox.com/scl/fi/pf2190qpfpiuo05mhcqmi/Adaptive-Guide-Erasure.pdf?rlkey=63s7ruwqxhrdsc4i603gjmsri&st=y79mr0ej&dl=0))*  

> [**Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation**](https://arxiv.org/abs/2410.15618),
> Tuan-Anh Bui, Long Vuong, Khanh Doan, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung
> *NeurIPS 2024 ([arXiv 2410.15618](https://arxiv.org/abs/2410.15618))*  

> [**Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts**](https://arxiv.org/abs/2403.12326),
> Tuan-Anh Bui, Khanh Doan, Trung Le, Paul Montague, Tamas Abraham, Dinh Phung
> *Preprint ([arXiv 2403.12326](https://arxiv.org/abs/2403.12326))*  

## Abstract

Diffusion models excel at generating visually striking content from text but can inadvertently produce undesirable or harmful content when trained on unfiltered internet data. A practical solution is to selectively removing target concepts from the model, but this may impact the remaining concepts. Prior approaches have tried to balance this by introducing a loss term to preserve neutral content or a regularization term to minimize changes in the model parameters, yet resolving this trade-off remains challenging. In this work, we propose to identify and preserving concepts most affected by parameter changes, termed as **adversarial concepts**. This approach ensures stable erasure with minimal impact on the other concepts. We demonstrate the effectiveness of our method using the Stable Diffusion model, showing that it outperforms state-of-the-art erasure methods in eliminating unwanted content while maintaining the integrity of other unrelated elements.

### Key Observations and Motivations

**(1)** Erasing different target concepts from text-to-image diffusion models leads to varying impacts on the remaining concepts. For instance, removing 'nudity' significantly affects related concepts like 'women' and 'men' but has minimal impact on unrelated concepts like 'garbage truck.'
**(2)** Neutral concepts lie in the middle of the sensitivity spectrum, suggesting that they do not adequately represent the model's capability to be preserved.
**(3)** Furthermore, the choice of concept to be preserved during erasure significantly impacts the model's generative capability; relying on neutral concepts, as in previous work, is not an optimal solution.
**(4)** This highlights the need for adaptive methods to identify and preserve the most sensitive concepts related to the target concept being erased, rather than relying on fixed neutral/generic concepts.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tuananhbui89.github.io/assets/img/AP_arXiv/SDv14/compare-ESD-nudity-ESD-garbage-truck-similarity_clip_nudity_20_side_by_side.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Analysis of the impact of erasing the target concept on the model's capability. The impact is measured by the difference of CLIP score $\delta(c)$ between the original model and the corresponding sanitized model.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tuananhbui89.github.io/assets/img/AP_arXiv/SDv14/compare_histogram_nudity_esd-nudity_20_CLIP_2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Sensitivity spectrum of concepts to the target concept "nudity". The histogram shows the distribution of the similarity score between outputs of the original model $\theta$ and the corresponding sanitized model $\theta_{c_e}'$ for each concept $c$ from the CLIP tokenizer vocabulary.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://tuananhbui89.github.io/assets/img/AP_arXiv/SDv14/SD-v1-4-ESD-garbage-truck-AE-similarity_clip_nudity_20_side_by_side.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Comparing the impact of erasing the same "garbage truck" to other concepts with different preserving strategies, including preserving a fixed concept such as " ", "lexus", or "road", and adaptively preserving the most sensitive concept found by our method.
</div>

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

We provide the evaluation results of our method and baselines in the `evaluation_folder` folder and the associated notebooks to reproduce the results in the paper.

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{bui2024erasing,
  title={Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation},
  author={Bui, Anh and Vuong, Long and Doan, Khanh and Le, Trung and Montague, Paul and Abraham, Tamas and Phung, Dinh},
  booktitle={NeurIPS},
  year={2024}
}

@article{bui2024removing,
  title={Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts},
  author={Bui, Anh and Doan, Khanh and Le, Trung and Montague, Paul and Abraham, Tamas and Phung, Dinh},
  journal={arXiv preprint arXiv:2403.12326},
  year={2024}
}
```

## References

This repository is based on the repository [Erasing Concepts from Diffusion Models](https://github.com/rohitgandikota/erasing)