import os
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from datasets import get_dataset
from functions.ckpt_util import download
import torchvision.utils as tvu
from guided_diffusion.script_util import create_model
from numpy import arange
import torchvision.transforms as transforms
from .SVD_utils import *
from torchvision.transforms.functional import to_pil_image
from transformers import BlipProcessor,  BlipModel
from sentence_transformers import SentenceTransformer
from transformers import logging

import torch.profiler


logging.set_verbosity_error()
def preprocess_for_blip(tensor):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    return transform(tensor)
def blip_init(device):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model     = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval().to(device)
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return processor, model, sentence_model
import torch
import torch.nn.functional as F

def get_brightness_scale_from_text(
    image_features,        
    anchor_texts,          
    blip_processor,        
    blip_model             
):
    device = image_features.device

    keys, texts = zip(*anchor_texts.items())  

    text_inputs = blip_processor(
        text=list(texts),
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        text_features = blip_model.get_text_features(**text_inputs)

    img_norm  = image_features / image_features.norm(dim=-1, keepdim=True)
    txt_norm  = text_features  / text_features.norm(dim=-1, keepdim=True)

    cosines = (img_norm @ txt_norm.T).squeeze(0)  # [sim_bright, sim_dark]

    probs   = F.softmax(cosines, dim=0)
    p0, p1  = probs.tolist() 

    score_info = {
        f"sim_{keys[0]}": float(cosines[0]),
        f"sim_{keys[1]}": float(cosines[1]),
        f"p_{keys[0]}":   float(probs[0]),
        f"p_{keys[1]}":   float(probs[1]),
        "scale":         float(probs[0])  
    }
    return score_info["scale"], score_info

def get_blip_description(blip_processor, blip_model,sentence_model ,image_tensor):
    device = next(blip_model.parameters()).device
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    image_pil = to_pil_image(image_tensor)

    inputs = blip_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = blip_model.get_image_features(pixel_values=inputs['pixel_values'])
    
    return image_features

def blip_getting_score(blip_processor, blip_model, _, image_tensor):
    anchors = {
        "bright": "High-contrast and high-light image.",
        "dark":   "Low-contrast and low-light image."
    }
    image_features = get_blip_description(blip_processor, blip_model, None, image_tensor)

    scale, score_info = get_brightness_scale_from_text(
        image_features,
        anchors,
        blip_processor,
        blip_model
    )
    return scale


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# Code based on DDNM
class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device(self.args.device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self):
        cls_fn = None
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        if self.config.model.use_fp16:
            model.convert_to_fp16()
        if self.config.model.class_cond:
            ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
            self.config.data.image_size, self.config.data.image_size))
            if not os.path.exists(ckpt):
                download(
                    'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                    self.config.data.image_size, self.config.data.image_size), ckpt)
        else:
            ckpt = os.path.join(self.args.exp, "logs/imagenet/256x256_diffusion_uncond.pt")
            if not os.path.exists(ckpt):
                download(
                    'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt',
                    ckpt)

        model.load_state_dict(torch.load(ckpt, map_location=self.device),strict=False)
        model.to(self.device)
        model.eval()

        print(
              f'{self.config.time_travel.T_sampling} sampling steps.',
              f'travel_length = {self.config.time_travel.travel_length},',
              f'travel_repeat = {self.config.time_travel.travel_repeat}.'
             )
        self.SVD_Diff(model,cls_fn)

    def SVD_Diff(self, model,cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            generator=g,
        )

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        pbar = tqdm.tqdm(val_loader)

        processor, blip_model, sentence_model = blip_init(self.device)
        for (input_y, classes),name in pbar:
            mean=torch.mean(input_y)
            print(mean)
            if mean<0.1:
                y=input_y
                indicator = config.data.INDICATOR
                outer_iterN = config.data.OUTER_ITERN
                inner_iterN = config.data.INNER_ITERN
                fy = y * (1 - y)
                x = y + indicator * fy
                Isinner = config.data.ISINNER
                for outer_iter in arange(outer_iterN):
                    x0 = x
                    if Isinner == 1:
                        for inner_iter in arange(inner_iterN[outer_iter]):
                            fx = x * (1 - x)
                            x = x0 + indicator * fx
                    else:
                        x = x0
                input_y=x

            H, W = input_y.shape[2:]
            input_y = check_image_size(input_y, 32)
            H_, W_ = input_y.shape[2:]
            name = name[0].split('/')[-1]
            input_y = input_y.to(self.device) 
            y_u, y_s, y_v = svd_decomposition(input_y)
            y  = input_y
            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            x = torch.randn(
                input_y.shape[0],
                config.data.channels,
                H_,
                W_,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                xs = [x]
                x0_preds = []
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 
                    
                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to(x.device)

                        et = model(xt,t)

                        if et.size(1) == 6:
                            et = et[:, :3]
                        
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_t = (x0_t - x0_t.min()) / (x0_t.max() - x0_t.min())
                        _,x0_t_s,_ = svd_decomposition((x0_t))
                        w = blip_getting_score(processor, blip_model, sentence_model, x0_t)
                        # x0_t_hat = svd_reconstruction(y_u, w*x0_t_s+(1-w/2)*y_s, y_v)
                        x0_t_hat = svd_reconstruction(y_u, w*x0_t_s+(1-w)*y_s, y_v)

                        # DDNM
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        eta = self.args.eta
                        
                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                        # Code form DDNM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)
                        
                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))   
                        
                x = xs[-2]
            x = torch.clamp(x, 0.0, 1.0)
            x = x[:, :, :H, :W]
            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{name}")
            )
            
# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
