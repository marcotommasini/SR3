
import torch
import torch.nn.functional as F
import sys


import os
import torch
import torch.nn as nn
import numpy as np
from functions import operations
from Data.dataset import get_mean_std, unzip_file, DataSet_Faces
from torch.utils.data import DataLoader




def sin_time_embeding(t, number_channels = 256):
  inv_freq = 1.0 / (10000 ** (torch.arange(0, number_channels, 2).float() / number_channels)).to("cuda")
  pos_enc_a = torch.sin(t.repeat(1, number_channels // 2) * inv_freq)
  pos_enc_b = torch.cos(t.repeat(1, number_channels // 2) * inv_freq)
  pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
  return pos_enc


class warmup_LR():
  def __init__(self, optmizer, intial_value, final_value, number_steps):
    self.opt = optmizer
    self.initial_value = intial_value
    self.final_value = final_value
    self.number_steps = number_steps
    self.values = torch.linspace(intial_value, final_value, number_steps)

  def linear(self, current_step):
    if current_step < self.number_steps:
      self.opt.param_groups[0]['lr'] = self.values[current_step]
    else:
      self.opt.param_groups[0]['lr'] = self.final_value
    return self.opt.param_groups[0]['lr']

	  
class beta_schedule():
  def __init__(self, beta_start, beta_end, number_timesteps):
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.number_timesteps = number_timesteps

  def linear(self):
    return torch.linspace(self.beta_start, self.beta_end, self.number_timesteps)

  def quadratic(self):
    return torch.linspace(self.beta_start**0.5, self.beta_end**0.5, self.number_timesteps) ** 2

  def sigmoid(self):
    betas = torch.linspace(-6, 6, self.number_timesteps)
    return (torch.sigmoid(betas) * (self.beta_end - self.beta_start)) + self.beta_start

  def cosine(self, s=0.008):
    
    steps = self.number_timesteps + 1
    x = torch.linspace(0, self.number_timesteps, steps)
    alphas_cumprod = torch.cos(((x / self.number_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, self.beta_start, self.beta_end)


def get_statistical_parameters(zipped_directory, images_directory, batch_size):
    unzip_file(zipped_directory, images_directory)
    dataset_faces = DataSet_Faces(images_directory)
    dataloader_faces = DataLoader(DataSet_Faces, batch_size = batch_size)
    return get_mean_std(dataloader_faces)

