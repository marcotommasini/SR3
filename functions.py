import os
from re import X
import torch
import torch.nn as nn
import argparse
import numpy as np
from utils import beta_schedule, sin_time_embeding, image_process
from functools import partial
from tqdm import tqdm


class operations:
    def __init__(self, args, number_noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, target_image_size = 128, device = "cuda"):
        self.number_noise_steps = args.number_noise_steps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.image_size = args.target_image_size
        self.device = args.device
        self.args = args

        schedule = beta_schedule(self.beta_start, self.beta_end, self.number_noise_steps)
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        if args.noise_schedule == "linear":
            self.beta = schedule.linear()
        elif args.noise_schedule == "quadratic":
            self.beta = schedule.quadratic()
        elif args.noise_schedule == "sigmoid":
            self.beta = schedule.sigmoid() 
        elif args.noise_schedule == "cosine":
            self.beta = schedule.cosine()
        
        beta_buffer = self.betas.detach().cpu().numpy()
        self.alpha = 1 - self.beta
        self.gamma = torch.cumprod(self.alpha, dim = 0)
        self.gamma_prev = np.append(1., self.gamma[:-1])
        self.sqrt_gamma_prev = np.sqrt(self.gamma_prev)


    def produce_noise(self, x, time_position):      #returns the noised image with a sample of a normal distribution
        part1 = torch.sqrt(self.gamma[time_position])[:, None, None, None]
        part2 = torch.sqrt(1 - self.gamma[time_position])[:, None, None, None]
        noise = torch.randn_like(x)
        return part1 * x + part2 * noise, noise
    
    def sample_image(self,model, x_low_res ):
        pass


    def train_model(self, model, dataloader, optmizer, loss, model_checkpoint = None):
        if self.args.use_checkpoints == "True" and model_checkpoint != None:  #Load the checkpoints of the model
            print("Using checkpoint")
            model.load_state_dict(model_checkpoint['model_state_dict'])
            optmizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
            epoch = model_checkpoint["epoch"]
        else:
            epoch = 0

        while epoch < self.args.number_epochs:
            list_losses = []
            with tqdm(dataloader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    optmizer.zero_grad()
                    
                    x_low = self.to_torch(data[0])

                   

                    x_upscaled = x_low  #Need to create an upscale function for the x in low resolution
                    x_high = self.to_torch(data[1])

                    t = torch.randint(1, self.number_noise_steps, (self.args.batch_size, )).to(self.device)

                    xt_noisy, normal_distribution = operations.produce_noise(x_high, t)
                    xt_noisy = self.to_torch(xt_noisy)
                    normal_distribution = self.to_torch(normal_distribution)
                    noise_level = torch.FloatTensor([self.gamma_prev[t+1]]).unsqueeze(-1).to(self.device)

                    sinusoidal_time_embeding = sin_time_embeding(noise_level).to(self.device) #This needs to be done because the UNET only accepts the time tensor when it is transformed
                    xt_cat = torch.cat((xt_noisy, x_upscaled), dim=1)

                    x_pred = model(xt_cat, sinusoidal_time_embeding).to(self.device)    #Predicted images from the UNET by inputing the image and the time without the sinusoidal embeding

                    Lsimple = loss(x_pred, normal_distribution).to(self.device)
                    
                    list_losses.append(Lsimple.item())
                    Lsimple.backward()
                    optmizer.step()

                    tepoch.set_postfix(loss=Lsimple.item())
                epoch += 1

                #Need to implement the function to save the checkpoint of the model


            
