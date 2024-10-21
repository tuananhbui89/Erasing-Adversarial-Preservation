from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
import clip
from PIL import Image
from torch.autograd import Variable
from utils_exp import get_prompt

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")


    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path,word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
    plt.close()


def get_models(config_path, ckpt_path, devices):
    model_orig = load_model_from_config(config_path, ckpt_path, devices[1])
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, devices[0])
    sampler = DDIMSampler(model)

    return model_orig, sampler_orig, model, sampler


def adversarial_prompt(model, model_orig, sampler, image_size, ddim_steps, pgd_num_steps, pgd_lr, init_value, emb_n, start_guidance, devices):
    """
    Learning adversarial prompt by Projected Gradient Descend algorithm
    Args: (Important param marked with * )
    * model: the model to be optimized
    model_orig: the foundation model, to be frozen 
    sampler: the sampler to be used for sampling, from the model
    image_size: the size of the image to be generated
    ddim_steps: the number of steps in the diffusion model
    start_guidance: the guidance to generate images for training
    
    * pgd_num_steps: the number of steps to optimize the prompt
    * pgd_lr: the learning rate for the prompt
    * init_value: the initial value of the prompt
    * emb_n: the conditional prompt
    
    """
    # Init 
    criteria = torch.nn.MSELoss()
    ddim_eta = 0

    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)
    
    # Step 1: Init learnable prompt with initial value 
    emb_prompt = Variable(init_value, requires_grad=True).to(devices[0])

    all_emb_prompt = []
    # Step 2: For loop to optimize prompt
    start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

    t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
    # time step from 1000 to 0 (0 being good)
    og_num = round((int(t_enc)/ddim_steps)*1000)
    og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])
    
    with torch.no_grad():
        z = quick_sample_till_t(emb_n.to(devices[0]), start_guidance, start_code, int(t_enc))

    for i in range(pgd_num_steps):

        # import pdb; pdb.set_trace()

        # track gradient 
        emb_prompt.requires_grad = True

        with torch.no_grad():
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_org = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_prompt.to(devices[1])) # ORG
            # e_org = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_n.to(devices[1])) # CHANGE HERE from emb_prompt to emb_n

        
        # get conditional score from model at time step t and image z
        e_n = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_prompt.to(devices[0]))

        # using DDIM inversion to project the x_t to x_0
        alpha_bar_t = sampler.ddim_alphas[int(t_enc)]
        z_n_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_n) / torch.sqrt(alpha_bar_t)
        z_org_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_org) / torch.sqrt(alpha_bar_t)

        # Step 2.2: Compute loss
        loss = criteria(z_n_pred.to(devices[0]), z_org_pred.to(devices[0]))

        # Step 2.3: Update prompt
        loss.backward()

        # get the gradient 
        grad = emb_prompt.grad
        if i == 0:
            print('norm of grad:', torch.norm(grad), torch.max(grad), torch.min(grad), torch.norm(emb_prompt), torch.max(emb_prompt), torch.min(emb_prompt))
        # update the prompt, maximizing the loss
        emb_prompt = emb_prompt + pgd_lr * grad / torch.norm(grad) * torch.norm(emb_prompt)

        # clip the prompt to be within the range
        # emb_prompt = torch.clamp(emb_prompt, -1, 1)

        # zero the gradient
        # emb_prompt.grad.zero_()
        model.zero_grad()

        # detach the prompt
        emb_prompt = emb_prompt.detach()
        all_emb_prompt.append(emb_prompt)
    

    return emb_prompt, all_emb_prompt


        


def train_prompt(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, seperator=None, image_size=512, ddim_steps=50):
    '''
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (noxattn, xattan).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    prompt, preserved = get_prompt(prompt)

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
        preserved_words = preserved.split(seperator)
        preserved_words = [word.strip() for word in preserved_words]
    else:
        words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train only qkv layers in x attention layers
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                print(name)
                parameters.append(param)
                # return_nodes[name] = name
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    
    # load clip model
    # clip_model, clip_preprocess = clip.load("ViT-B/32", devices[0])

    # import pdb; pdb.set_trace()
    def decode_and_extract_image(model_orig, z):
        x = model_orig.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b (c h) w')
        image = clip_preprocess(Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8)))
        with torch.no_grad():
            image_features = clip_model.encode_image(image.unsqueeze(0).to(devices[0]))
        return image_features
    
    def decode_and_save_image(model_orig, z, path):
        x = model_orig.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()

    def extract_text(text):
        assert isinstance(text, str)
        text = [text]
        text = clip.tokenize(text).to(devices[0])
        with torch.no_grad():
            text_features = clip_model.encode_text(text)
        return text_features

    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    name = f'compvis-adversarial-pgd-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}-lr_{lr}-info_{args.info}'
    models_path = args.models_path
    os.makedirs(f'evaluation_folder/{name}', exist_ok=True)
    os.makedirs(f'{models_path}/{name}', exist_ok=True)


    # TRAINING CODE
    pbar = tqdm(range(iterations))

    # Create a dictionary to store the prompt 
    # each word has a prompt with size prompt_size

    def create_prompt(word):
        prompt = f'{word}'
        emb = model.get_learned_conditioning([prompt])
        init = emb
        # init = emb + torch.randn_like(emb) * torch.norm(emb) * 0.001
        return init

    prompt_dict = {}    
    for word in words:
        prompt_dict[word] = create_prompt(word)

    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])    


    save_dict = dict()


    for i in pbar:
        word = random.sample(words,1)[0]
        retained_word = random.sample(preserved_words,1)[0]

        # 
        prompt_0 = ''
        prompt_r = f'{retained_word}'
        prompt_n = f'{word}'

        # get text embeddings for unconditional and conditional prompts
        emb_0 = model.get_learned_conditioning([prompt_0])
        # emb_r = model.get_learned_conditioning([prompt_r])
        emb_n = model.get_learned_conditioning([prompt_n])

        opt.zero_grad()
        model.zero_grad()
        model_orig.zero_grad()

        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate an image with the concept
            z = quick_sample_till_t(emb_n.to(devices[0]), start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0_org = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            e_n_org = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_n.to(devices[1]))
            e_p_org = model_orig.apply_model(z.to(devices[1]), t_enc_ddpm.to(devices[1]), prompt_dict[word].to(devices[1]))

        # breakpoint()
        # get conditional score
        e_p = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), prompt_dict[word].to(devices[0]))
        e_n_wo_prompt = model.apply_model(z.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_n.to(devices[0]))

        e_0_org.requires_grad = False
        e_n_org.requires_grad = False
        e_p_org.requires_grad = False

        # using DDIM inversion to project the x_t to x_0
        # check that the alphas is in descending order
        assert torch.all(sampler.ddim_alphas[:-1] >= sampler.ddim_alphas[1:])
        alpha_bar_t = sampler.ddim_alphas[int(t_enc)]
        z_p_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_p) / torch.sqrt(alpha_bar_t)
        z_n_wo_prompt_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_n_wo_prompt) / torch.sqrt(alpha_bar_t)

        z_n_org_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_n_org) / torch.sqrt(alpha_bar_t)
        z_0_org_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_0_org) / torch.sqrt(alpha_bar_t)
        z_p_org_pred = (z - torch.sqrt(1 - alpha_bar_t) * e_p_org) / torch.sqrt(alpha_bar_t)

        # First stage, optimizing additional prompt

        # for erased concepts, output aligns with target concept with or without prompt
        loss = 0
        if args.erase_wo_prompt != 0:
            loss += args.erase_wo_prompt * criteria(z_n_wo_prompt_pred.to(devices[0]), z_0_org_pred.to(devices[0]) - (negative_guidance * (z_n_org_pred.to(devices[0]) - z_0_org_pred.to(devices[0]))))
        if args.erase_w_prompt != 0:
            loss += args.erase_w_prompt * criteria(z_p_pred.to(devices[0]), z_p_org_pred.to(devices[0]))
        
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()
        opt.zero_grad()

        # learn the prompt
        emb_prompt, _ = adversarial_prompt(model, model_orig, sampler, image_size, ddim_steps, args.pgd_num_steps, args.pgd_lr, prompt_dict[word], emb_n, start_guidance, devices)
        prompt_dict[word] = emb_prompt
        
        if word not in save_dict:
            save_dict[word] = []

            _, all_emb_prompt = adversarial_prompt(model, model_orig, sampler, image_size, ddim_steps, 2*args.pgd_num_steps, args.pgd_lr/2, prompt_dict[word], emb_n, start_guidance, devices)

            z = quick_sample_till_t(emb_n.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
            decode_and_save_image(model_orig, z, f'evaluation_folder/{name}/original_prompt_{word}_first.png')
            # generate an image with emb_prompt 
            for ip, eval_emb_prompt in enumerate(all_emb_prompt):
                print('norm of diff:', torch.norm(all_emb_prompt[ip] - all_emb_prompt[ip-1]))
                z = quick_sample_till_t(eval_emb_prompt.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                decode_and_save_image(model_orig, z, f'evaluation_folder/{name}/adversarial_prompt_{word}_{ip}_first.png')

        if (i+1) % args.save_freq == 0:
            _, all_emb_prompt = adversarial_prompt(model, model_orig, sampler, image_size, ddim_steps, 2*args.pgd_num_steps, args.pgd_lr/2, prompt_dict[word], emb_n, start_guidance, devices)

            z = quick_sample_till_t(emb_n.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
            decode_and_save_image(model_orig, z, f'evaluation_folder/{name}/original_prompt_{word}.png')
            # generate an image with emb_prompt 
            for ip, eval_emb_prompt in enumerate(all_emb_prompt):
                print('norm of diff:', torch.norm(all_emb_prompt[ip] - all_emb_prompt[ip-1]))
                z = quick_sample_till_t(eval_emb_prompt.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                decode_and_save_image(model_orig, z, f'evaluation_folder/{name}/adversarial_prompt_{word}_{ip}.png')

        # exit()

        # save checkpoint and loss curve
        if (i+1) % 500 == 0 and i+1 != iterations and i+1>= 500:
            # save_model(model, name, i-1, save_compvis=True, save_diffusers=False)
            save_model(model, name, None, models_path=models_path, save_compvis=True, save_diffusers=False)

            # save the prompt
            torch.save(prompt_dict, f'{models_path}/{name}/prompt_dict.pt')
            
        if i % 100 == 0:
            save_history(losses, name, word_print, models_path=models_path)

    model.eval()

    save_model(model, name, None, models_path=models_path, save_compvis=True, save_diffusers=False, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print, models_path=models_path)
    # save the prompt
    torch.save(prompt_dict, f'{models_path}/{name}/prompt_dict.pt')
    
def save_model(model, name, num, models_path, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    # SAVE MODEL

#     PATH = f'{FOLDER}/{model_type}-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{neg_guidance}-iter_{i+1}-lr_{lr}-startmodel_{start_model}-numacc_{numacc}.pt'

    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'

    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device )

def save_history(losses, name, word_print, models_path):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses,f'{folder_path}/loss.png' , word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--erase_wo_prompt', help='hyper-param for erasing wo prompt term', type=float, required=False, default=1)
    parser.add_argument('--erase_w_prompt', help='hyper-param for erasing with prompt term', type=float, required=False, default=0.01)
    parser.add_argument('--prompt_size', help='size of prompt', type=int, required=False, default=1)
    parser.add_argument('--prompt_mode', help='method of prompting', type=str, required=True)
    parser.add_argument('--prompt_optk', help='time to optimize prompt', type=int, required=False, default=2)
    parser.add_argument('--prompt_lr', help='learning rate for prompt', type=float, required=False, default=1e-5)
    parser.add_argument('--save_freq', help='frequency to save data, per iteration * prompt_optk', type=int, required=False, default=10)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')
    parser.add_argument('--pgd_num_steps', help='number of steps to optimize prompt', type=int, required=False, default=100)
    parser.add_argument('--pgd_lr', help='learning rate for prompt', type=float, required=False, default=1e-5)

    args = parser.parse_args()
    
    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_prompt(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps)
