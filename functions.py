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
import torchvision.transforms as transforms
import metrics



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
        self.posterior_mean_coef1 = self.to_torch(self.posterior_mean_coef1).unsqueeze(-1)
        self.posterior_mean_coef2 = self.to_torch(self.posterior_mean_coef2).unsqueeze(-1)

        self.counter_iterations = 0


    def produce_noise(self, x_start, continuous_sqrt_gamma_prev):      #returns the noised image with a sample of a normal distribution
        noise = torch.randn_like(x_start)
        part1 = continuous_sqrt_gamma_prev[:, None, None, None]
        part2 = 1 - (continuous_sqrt_gamma_prev**2).sqrt()[:, None, None, None]
        return part1 * x_start + part2 * noise, noise



    def train_model(self, model, dataloader, optmizer, loss, model_checkpoint = None):
        print("Training started")
        print("Length: ", len(dataloader))

        model.train()
        LRS = warmup_LR(optmizer, self.args.initial_learning_rate, self.args.final_learning_rate, number_steps=250)
        
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
                self.learning_rate = LRS.linear(epoch * len(dataloader) + i) #updating the value of the learning rate

                optmizer.zero_grad()
                
                x_low = data[1].to(self.device)
                x_high = data[0].to(self.device)

                t = np.random.randint(1, self.args.number_noise_steps)

                continuous_sqrt_gamma_prev = np.random.uniform(self.sqrt_gamma_prev[t-1].cpu().numpy(), self.sqrt_gamma_prev[t].cpu().numpy(), size = self.args.batch_size) 
                continuous_sqrt_gamma_prev = self.to_torch(continuous_sqrt_gamma_prev)

                xt_noisy, normal_distribution = self.produce_noise(x_high, continuous_sqrt_gamma_prev)

                normal_distribution = normal_distribution.to(self.device)
                continuous_sqrt_gamma_prev = continuous_sqrt_gamma_prev.unsqueeze(-1)
                sinusoidal_time_embeding = sin_time_embeding(continuous_sqrt_gamma_prev, device=self.device) #This needs to be done because the UNET only accepts the time tensor when it is transformed

                xt_cat = torch.cat((x_low, xt_noisy), dim=1)
                x_pred = model(xt_cat, sinusoidal_time_embeding)    #Predicted images from the UNET by inputing the image and the time without the sinusoidal embeding
                x_pred = x_pred.to(self.device)

                Lsimple = loss(x_pred, normal_distribution).to(self.device)
                print(Lsimple)
                print(Lsimple.size())
                import sys
                sys.exit()
                list_losses.append(Lsimple.item())
                Lsimple.backward()
                optmizer.step()
            try:
                if epoch % 5 == 0:
                    image_sampled = self.p_sample(model, x_low)[0]
                    image = metrics.tensor2img(image_sampled)
                    metrics.save_img(image, "/content/drive/MyDrive/SR3/checkpoint_directory/fudeu_vida.png")
            except:
                pass 
            epoch += 1
            print("LR: ", self.learning_rate)

            EPOCH = epoch
            PATH = self.args.checkpoint_directory + "/checkpoint.pt"      
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optmizer.state_dict(),
                'LRSeg': self.learning_rate,
                }, PATH)
            print("checkpoint saved")
            print("Mean: ", np.mean(list_losses))


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

                alpha_buffer = self.alpha[i][:,None, None, None]
                gamma_buffer = self.gamma[i][:, None, None, None]
                beta_buffer = self.beta[i][:, None, None, None]
                gamma_prev_buffer = self.gamma_prev[i][:, None, None, None]
                sqrt_recip_alphas_cumprod_buffer = self.sqrt_recip_alphas_cumprod[i]
                sqrt_recipm1_alphas_cumprod_buffer = self.sqrt_recipm1_alphas_cumprod[i]
                posterior_mean_coef1_buffer = self.posterior_mean_coef1[i]
                posterior_mean_coef2_buffer = self.posterior_mean_coef2[i]
                log_posterior_variance_buffer = self.to_torch(self.log_posterior_variance[i])

                #Calculating the mean of the posterior
                noise_level = torch.FloatTensor([self.sqrt_gamma_prev[i]]).repeat(batch_size, 1).to(self.args.device)
                sinusoidal_noise_embeding = sin_time_embeding(noise_level, device=self.device)
                pred_noise = model(x_cat, sinusoidal_noise_embeding)

                x_recon = sqrt_recip_alphas_cumprod_buffer* x_noise - sqrt_recipm1_alphas_cumprod_buffer * pred_noise
                x_recon.clamp_(-1., 1.)
                model_mean =  posterior_mean_coef1_buffer * x_recon + posterior_mean_coef2_buffer * x_noise

                #Calculating the variance of the posterior
                model_variance = (0.5 * log_posterior_variance_buffer).exp()

                xtm = model_mean + model_variance * z

                x_noise = xtm
        
        model.train()
        return x_noise
    

    def p_sample(self, model, condition_x=None):
        shape = condition_x.shape
        img = torch.randn(shape, device=self.device)
        with torch.no_grad():
            for t in tqdm(reversed(range(0, self.args.number_noise_steps)), desc='sampling loop time step', total=self.args.number_noise_steps):
                batch_size = self.args.batch_size
                noise_level = torch.FloatTensor([self.sqrt_gamma_prev[t]]).repeat(batch_size, 1).to(self.device)
                noise_level = sin_time_embeding(noise_level, device=self.device)
            
                x_cat = torch.cat([condition_x, img], dim=1)
                pred_noise = model(x_cat, noise_level)
                x_recon = self.sqrt_recip_alphas_cumprod[t] * img - self.sqrt_recipm1_alphas_cumprod[t] * pred_noise

                x_recon.clamp_(-1., 1.)
                model_mean = self.posterior_mean_coef1[t] * x_recon + self.posterior_mean_coef2[t] * img
                log_posterior_variance_buffer = self.to_torch(self.log_posterior_variance)
                posterior_log_variance = log_posterior_variance_buffer[t]
                
                normal_noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
                img = model_mean + normal_noise * (0.5 * posterior_log_variance).exp()
                result = img
        return result




