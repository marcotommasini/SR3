import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from utils import beta_schedule, sin_time_embeding, warmup_LR



class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        args,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.device = args.device
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    
        to_torch = partial(torch.tensor, dtype=torch.float32, device=args.device)

        schedule = beta_schedule(args.beta_start, args.beta_end, args.number_noise_steps)


        if args.noise_schedule == "linear":
            betas = schedule.linear()
        elif args.noise_schedule == "quadratic":
            betas = schedule.quadratic()
        elif args.noise_schedule == "sigmoid":
            betas = schedule.sigmoid() 
        elif args.noise_schedule == "cosine":
            betas = schedule.cosine()
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


    def p_sample(self,condition_x=None):
        device = self.betas.device
        shape = condition_x.shape
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            batch_size = img.shape[0]
            noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(condition_x.device)
            noise_level = sin_time_embeding(noise_level, device=self.device)
        
            x_cat = torch.cat([condition_x, img], dim=1)
            pred_noise = self.denoise_fn(x_cat, noise_level)
            x_recon = self.sqrt_recip_alphas_cumprod[t] * img - self.sqrt_recipm1_alphas_cumprod[t] * pred_noise

            
            x_recon.clamp_(-1., 1.)
            model_mean = self.posterior_mean_coef1[t] * x_recon + self.posterior_mean_coef2[t] * img
            posterior_log_variance = self.posterior_log_variance_clipped[t]
            
            normal_noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            img = model_mean + normal_noise * (0.5 * posterior_log_variance).exp()
            result = img
        return result


    def super_resolution(self, x_in):
        return self.p_sample(x_in)
