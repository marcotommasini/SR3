import os
from unittest import loader
import torch
import torch.nn as nn
import argparse
import numpy as np
from utils import beta_schedule, sin_time_embeding, warmup_LR
from functools import partial
from tqdm import tqdm
from dataset import image_process
import time

class operations:
    def __init__(self, args):
        self.number_noise_steps = args.number_noise_steps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.image_size = args.target_image_size
        self.device = args.device
        self.args = args
        self.learning_rate = 0
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=args.device)

        schedule = beta_schedule(self.beta_start, self.beta_end, self.number_noise_steps)


        if args.noise_schedule == "linear":
            betas = schedule.linear()
        elif args.noise_schedule == "quadratic":
            betas = schedule.quadratic()
        elif args.noise_schedule == "sigmoid":
            betas = schedule.sigmoid() 
        elif args.noise_schedule == "cosine":
            betas = schedule.cosine()

        

        self.beta = betas.detach().cpu().numpy()
        self.alpha = 1 - self.beta
        self.gamma =  np.cumprod(self.alpha, axis = 0)
        self.gamma_prev = np.append(1., self.gamma[:-1])
        self.sqrt_gamma_prev = np.sqrt(np.append(1., self.gamma[:-1]))
        self.posterior_mean_coef1 = self.beta * np.sqrt(self.sqrt_gamma_prev) / (1. - self.gamma)
        self.posterior_mean_coef2 = (1. - self.sqrt_gamma_prev) * np.sqrt(self.alpha) / (1. - self.gamma)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.gamma)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.gamma - 1)

        self.posterior_variance = self.beta * (1. - self.gamma_prev)/(1. - self.gamma)   #This implementation will not use the standard variance for the posterior a change in variance can be done in order to test is's importance in the model
        self.log_posterior_variance = np.log(np.maximum(self.posterior_variance, 1e-20))
        
        #Transforming into tensor float32
        self.beta = self.to_torch(self.beta).unsqueeze(-1)
        self.alpha = self.to_torch(self.alpha).unsqueeze(-1)
        self.gamma = self.to_torch(self.gamma).unsqueeze(-1)
        self.gamma_prev = self.to_torch(self.gamma_prev).unsqueeze(-1)
        self.sqrt_gamma_prev = self.to_torch(self.sqrt_gamma_prev).unsqueeze(-1)
        self.sqrt_recip_alphas_cumprod = self.to_torch(self.sqrt_recip_alphas_cumprod).unsqueeze(-1)
        self.sqrt_recipm1_alphas_cumprod = self.to_torch(self.sqrt_recipm1_alphas_cumprod).unsqueeze(-1)
        posterior_mean_coef1 = self.to_torch(self.posterior_mean_coef1).unsqueeze(-1)
        self.posterior_mean_coef2 = self.to_torch(self.posterior_mean_coef2).unsqueeze(-1)

        self.counter_iterations = 0


    def produce_noise(self, x, time_position):      #returns the noised image with a sample of a normal distribution
        part1 = torch.sqrt(self.gamma[time_position])[:, None, None, None]
        part2 = torch.sqrt(1 - self.gamma[time_position])[:, None, None, None]
        noise = torch.randn_like(x)
        return part1 * x + part2 * noise, noise



    def train_model(self, model, dataloader, optmizer, loss, model_checkpoint = None):
        print("Training started")
        model.train()
        LRS = warmup_LR(optmizer, self.args.initial_learning_rate, self.args.final_learning_rate, number_steps=1000)
        
        if self.args.use_checkpoints == "True" and model_checkpoint != None:  #Load the checkpoints of the model
            print("Using checkpoint")
            model.load_state_dict(model_checkpoint['model_state_dict'])
            optmizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
            epoch = model_checkpoint["epoch"]
        else:
            epoch = 0

        while epoch < self.args.number_epochs:
            print("epoch: ", epoch)
            list_losses = []
            for i, data in tqdm(enumerate(dataloader)):
                self.learning_rate = LRS.linear(i) #updating the value of the learning rate

                optmizer.zero_grad()
                
                x_low = data[1].to(self.device)
                x_high = data[0].to(self.device)

                t = torch.randint(1, self.number_noise_steps, (self.args.batch_size, )).to(self.device)

                xt_noisy, normal_distribution = self.produce_noise(x_high, t)
                xt_noisy = xt_noisy.to(self.device)
                normal_distribution = normal_distribution.to(self.device)

                noise_level = self.gamma_prev[t].unsqueeze(-1)   #This model does not use t for the embeddin, they use a variation of gamma
                
                sinusoidal_time_embeding = sin_time_embeding(noise_level, device=self.device) #This needs to be done because the UNET only accepts the time tensor when it is transformed

                xt_cat = torch.cat((xt_noisy, x_low), dim=1)
                x_pred = model(xt_cat, sinusoidal_time_embeding)    #Predicted images from the UNET by inputing the image and the time without the sinusoidal embeding
                x_pred = x_pred.to(self.device)

                Lsimple = loss(x_pred, normal_distribution).to(self.device)
                
                list_losses.append(Lsimple.item())
                Lsimple.backward()
                optmizer.step()

            epoch += 1

            EPOCH = epoch
            PATH = self.args.checkpoint_directory + "/checkpoint.pt"      
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optmizer.state_dict(),
                'LRSeg': self.learning_rate,
                }, PATH)
            print("checkpoint saved")


    def sample_image(self, model, x_low_res):
        print("Start sampling")
        batch_size, channel_size, y_size, x_size = x_low_res.size()
        model.eval()

        x_noise = torch.randn(batch_size, channel_size, self.image_size, self.image_size)
        x_noise = x_noise.to(self.device)

        x_upsample = x_low_res.to(self.device)

        
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.number_noise_steps))):
                x_cat = torch.cat([x_noise, x_upsample], 1)

                noise_level = torch.FloatTensor([self.sqrt_gamma_prev[i]]).repeat(batch_size, 1).to(self.args.device)
                
                if i == 0:
                    z = torch.zeros(x_noise.size())
                else:
                    z = torch.randn_like(x_noise)

                # print(self.alpha.size())
                # print(self.beta.size())
                # print(self.gamma.size())
                # import sys
                # sys.exit()
                alpha_buffer = self.alpha[i][:,None, None, None]
                gamma_buffer = self.gamma[i][:, None, None, None]
                beta_buffer = self.beta[i][:, None, None, None]
                gamma_prev_buffer = self.gamma_prev[i][:, None, None, None]
                sqrt_recip_alphas_cumprod_buffer = self.sqrt_recip_alphas_cumprod[i]
                sqrt_recipm1_alphas_cumprod_buffer = self.sqrt_recipm1_alphas_cumprod[i][:, None, None, None]
                posterior_mean_coef2_buffer = self.posterior_mean_coef1[i]
                posterior_mean_coef2_buffer = self.posterior_mean_coef2[i]
                log_posterior_variance_buffer = self.to_torch(self.log_posterior_variance[i])

                #Calculating the mean of the posterior
                noise_level = torch.FloatTensor([self.sqrt_gamma_prev[i]]).repeat(batch_size, 1).to(self.args.device)
                sinusoidal_noise_embeding = sin_time_embeding(noise_level, device=self.device)
                pred_noise = model(x_cat, sinusoidal_noise_embeding)

                x_recon = sqrt_recip_alphas_cumprod_buffer* x_noise - sqrt_recipm1_alphas_cumprod_buffer * pred_noise
                x_recon.clamp_(-1., 1.)
                model_mean =  posterior_mean_coef2_buffer * x_recon + posterior_mean_coef2_buffer * x_noise

                #Calculating the variance of the posterior
                model_variance = (0.5 * log_posterior_variance_buffer).exp()

                xtm = model_mean + model_variance * z

                x_noise = xtm

            x_noise = (x_noise.clamp(-1, 1) + 1) / 2
            x_noise = (x_noise * 255).type(torch.uint8)
        
        model.train()
        return x_noise




