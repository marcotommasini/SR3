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

        schedule = beta_schedule(self.beta_start, self.beta_end, self.number_noise_steps)


        if args.noise_schedule == "linear":
            self.beta = schedule.linear()
        elif args.noise_schedule == "quadratic":
            self.beta = schedule.quadratic()
        elif args.noise_schedule == "sigmoid":
            self.beta = schedule.sigmoid() 
        elif args.noise_schedule == "cosine":
            self.beta = schedule.cosine()
        
        self.alpha = 1 - self.beta
        self.gamma = torch.cumprod(self.alpha, dim = 0).to(self.device)
        self.gamma_prev = torch.tensor(np.append(1., self.gamma[:-1].cpu().detach().numpy()), dtype=torch.float32).to(self.device)
        self.sqrt_gamma_prev = torch.sqrt(self.gamma_prev)

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
            PATH = self.args.checkpoint_directory       
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
                x_cat = torch.cat([x_noise, x_upsample])

                t = (torch.ones(batch_size, dtype= torch.float32) * i).to(self.device)
                
                if i == 0:
                    z = torch.zeros(x_noise.size())
                else:
                    z = torch.randn_like(x_noise)
                
                posterior_variance = self.beta * (1. - self.gamma_prev)/(1. - self.gamma)   #This implementation will not use the standard variance for the posterior a change in variance can be done in order to test is's importance in the model
                log_var_arg_2 = torch.ones(posterior_variance.size(), dtype=torch.float32).to(self.device)
                log_posterior_variance = torch.log(torch.maximum(posterior_variance, log_var_arg_2))
                model_variance = torch.exp(0.5 * log_posterior_variance)

                
                alpha_buffer = self.alpha[t][:, None, None, None]
                gamma_buffer = self.gamma_buffer[t][:, None, None, None]
                beta_buffer = self.beta[t][:, None, None, None]

                noise_level = self.gamma_prev[t].unsqueeze(-1)

                sinusoidal_noise_embeding = sin_time_embeding(noise_level, device=self.device)

                pred_noise = model(x_cat, sinusoidal_noise_embeding)
                
                part2 = ((1 - alpha_buffer)/(torch.sqrt(1 - gamma_buffer))) * pred_noise
                xtm = ((1/torch.sqrt(alpha_buffer)) * (x_noise - part2)) + model_variance * z

                x_noise = xtm

            x_noise = (x_noise.clamp(-1, 1) + 1) / 2
            x_noise = (x_noise * 255).type(torch.uint8)
        
        model.train()
        return x_noise




