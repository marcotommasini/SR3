import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from utils import beta_schedule, sin_time_embeding, image_process, warmup_LR
from functools import partial
from tqdm import tqdm


class operations:
    def __init__(self, args, number_noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, target_image_size = 128):
        self.number_noise_steps = args.number_noise_steps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.image_size = args.target_image_size
        self.device = args.device
        self.args = args
        self.learning_rate = 0

        schedule = beta_schedule(self.beta_start, self.beta_end, self.number_noise_steps)
        self.to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.IP = image_process()


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

        self.counter_iterations = 0

        self.alpha = self.to_torch(self.alpha)
        self.gamma = self.to_torch(self.gamma)
        self.gamma_prev = self.to_torch(self.gamma_prev)
        self.sqrt_gamma_prev = self.to_torch(self.sqrt_gamma_prev)


    def produce_noise(self, x, time_position):      #returns the noised image with a sample of a normal distribution
        part1 = torch.sqrt(self.gamma[time_position])[:, None, None, None]
        part2 = torch.sqrt(1 - self.gamma[time_position])[:, None, None, None]
        noise = torch.randn_like(x)
        return part1 * x + part2 * noise, noise



    def train_model(self, model, dataloader, optmizer, loss, model_checkpoint = None):
        LRS = warmup_LR(optmizer, self.args.initial_learning_rate, self.args.final_learning_rate, number_steps=1000)
        
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
                    self.learning_rate = LRS.linear(i) #updating the value of the learning rate

                    tepoch.set_description(f"Epoch {epoch}")
                    optmizer.zero_grad()
                    
                    x_low = self.to_torch(data[0])
                    x_high = self.to_torch(data[1])
                    x_upscaled = self.IP.image_upscale(x_low, x_high.size())  

                    t = torch.randint(1, self.number_noise_steps, (self.args.batch_size, )).to(self.device)

                    xt_noisy, normal_distribution = operations.produce_noise(x_high, t)
                    xt_noisy = self.to_torch(xt_noisy)
                    normal_distribution = self.to_torch(normal_distribution)

                    noise_level = self.to_torch([self.gamma_prev[t+1]]).unsqueeze(-1)   #This model does not use t for the embeddin, they use a variation of gamma
                    
                    sinusoidal_time_embeding = self.to_torch(sin_time_embeding(noise_level)) #This needs to be done because the UNET only accepts the time tensor when it is transformed

                    xt_cat = torch.cat((xt_noisy, x_upscaled), dim=1)

                    x_pred = model(xt_cat, sinusoidal_time_embeding)    #Predicted images from the UNET by inputing the image and the time without the sinusoidal embeding
                    x_pred = self.to_torch(x_pred)

                    Lsimple = loss(x_pred, normal_distribution).to(self.device)
                    
                    list_losses.append(Lsimple.item())
                    Lsimple.backward()
                    optmizer.step()

                    tepoch.set_postfix(loss=Lsimple.item())

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
        x_noise = self.to_torch(x_noise)

        x_upsample = self.IP.image_upscale(x_low_res, x_noise.size())
        x_upsample = self.to_torch(x_upsample)

        
        with torch.no_grad():
            for i in tqdm(reversed(range(1, self.number_noise_steps))):
                x_cat = torch.cat([x_noise, x_upsample])

                t = self.to_torch((torch.ones(batch_size) * i))
                
                if i == 0:
                    z = torch.zeros(x_noise.size())
                else:
                    z = torch.randn_like(x_noise)
                
                posterior_variance = self.beta * (1. - self.gamma_prev)/(1. - self.gamma)   #This implementation will not use the standard variance for the posterior a change in variance can be done in order to test is's importance in the model
                log_var_arg_2 = self.to_torch(torch.ones(posterior_variance.size()))
                log_posterior_variance = torch.log(torch.maximum(posterior_variance, log_var_arg_2))
                model_variance = torch.exp(0.5 * log_posterior_variance)

                
                alpha_buffer = self.alpha[t][:, None, None, None]
                gamma_buffer = self.gamma_buffer[t][:, None, None, None]
                beta_buffer = self.beta[t][:, None, None, None]

                noise_level = self.to_torch([self.gamma_prev[t]]).unsqueeze(-1)

                sinusoidal_noise_embeding = self.to_torch(sin_time_embeding(noise_level))

                pred_noise = model(x_cat, sinusoidal_noise_embeding)
                
                part2 = ((1 - alpha_buffer)/(torch.sqrt(1 - gamma_buffer))) * pred_noise
                xtm = ((1/torch.sqrt(alpha_buffer)) * (x_noise - part2)) + model_variance * z

                x_noise = xtm

            x_noise = (x_noise.clamp(-1, 1) + 1) / 2
            x_noise = (x_noise * 255).type(torch.uint8)
        
        model.train()
        return x_noise




