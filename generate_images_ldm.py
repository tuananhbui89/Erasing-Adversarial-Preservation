import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import pandas as pd

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_model_from_config_compvis(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--prompts_path', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=True)
    parser.add_argument('--from_case', help='continue generating from case_number', type=int, required=False, default=0)
    parser.add_argument('--to_case', help='continue generating to to_case number', type=int, required=False, default=-1)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')


    args = parser.parse_args()

    seed_everything(20231001)
    models_path = args.models_path
    
    if args.model_name == 'SD-v1-4':
        print('Using stable diffusion v1-4 model')
    else:
        try:
            model_path = f'{models_path}/{args.model_name}/{args.model_name}.pt'
        except Exception as e:
            print(f'Model path is not valid, please check the file name and structure: {e}')
            exit()

    config = OmegaConf.load(f"{args.config_path}")

    if args.ckpt_path:
        model = load_model_from_config(config, args.ckpt_path)
        print('Loading model from ckpt_path')
    else:
        model = load_model_from_config_compvis(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if args.dpm_solver:
        print("Using DPM solver sampler...")
        exit()
        sampler = DPMSolverSampler(model)
    elif args.plms:
        print("Using PLMS sampler...")
        exit()
        sampler = PLMSSampler(model)
    else:
        print("Using DDIM sampler...")
        sampler = DDIMSampler(model)

    df = pd.read_csv(args.prompts_path)
    prompts_name = args.prompts_path.split('/')[-1].split('.')[0]

    folder_path = f'{args.save_path}/{args.model_name}'
    subfolder_path = f'{folder_path}/ldm-{prompts_name}'

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(subfolder_path, exist_ok=True)

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "StableDiffusionV1"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # Convert to diffuser model
    # print('Converting model to diffuser model')
    # from convertModels import savemodelDiffusers
    # compvis_config_file = 'configs/stable-diffusion/v1-inference.yaml'
    # diffusers_config_path = '../Better_Erasing/models/erase/config.json'
    # name = 'adversarial-gumbel-word_imagenette_v1_wo-method_xattn-sg_3-ng_1-iter_1000-lr_1e-05-info_gumbel_lr_1e-2_temp_1_hard_1_num_200_update_-1_timestep_0_multi_2_kclosest_1000'
    # savemodelDiffusers(name, compvis_config_file, diffusers_config_path, device=device )
    # print('Model converted')
    # exit()

    precision_scope = autocast if args.precision=="autocast" else nullcontext
    with torch.no_grad():
        # with precision_scope("cuda"):
        #     with model.ema_scope():
                tic = time.time()

                for _, row in df.iterrows():

                    prompts = [str(row.prompt)]*args.num_samples
                    seed = row.evaluation_seed
                    case_number = row.case_number
                    
                    print('case_number:', row.case_number, 'prompt:', row.prompt, 'from_case:', args.from_case, 'to_case:', args.to_case)

                    if case_number<args.from_case:
                        continue
                    
                    if args.to_case != -1 and case_number > args.to_case:
                        break

                    generator = torch.cuda.manual_seed(seed)

                    latents = torch.randn(
                        [args.num_samples, args.C, args.H // args.f, args.W // args.f],
                        device=device,
                        generator=generator,
                    )
                    
                    uc = None
                    if args.scale != 1.0:
                        uc = model.get_learned_conditioning(args.num_samples * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    
                    c = model.get_learned_conditioning(prompts)
                    shape = [args.C, args.H // args.f, args.W // args.f]
                    samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=args.num_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=args.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=args.ddim_eta,
                                                        x_T=latents)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    for num, x_sample in enumerate(x_checked_image_torch):
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        # img = put_watermark(img, wm_encoder)
                        img.save(f"{subfolder_path}/{case_number}_{num}.png")

                toc = time.time()


if __name__ == "__main__":
    main()
