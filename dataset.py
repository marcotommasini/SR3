from locale import normalize
import os
import torch
from torchvision.io import read_image
from torch.utils import data
from torchvision.transforms.functional import convert_image_dtype
from torchvision import transforms
import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import zipfile
from functools import partial
from tqdm import tqdm


class Dataset(data.Dataset):
    def __init__(self, dataset_directory = None, image_dimensions = (128,128), downsample_dimensions = (16,16), transform_lists = None):
        super().__init__()
        self.dataset_directory = dataset_directory
        self.image_dimensions = image_dimensions
        self.downsample_dimensions = downsample_dimensions
        self.list_images = os.listdir(dataset_directory)

        self.transforms_HIGH = transform_lists[0]
        self.transforms_LOW = transform_lists[1]

        self.transforms_HIGH.insert(0,transforms.ToTensor())
        self.transforms_LOW.append(transforms.Resize(self.downsample_dimensions))
        

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        target_path = os.path.join(self.dataset_directory, self.list_images[index])
        
        apply_HIGH = transforms.Compose(self.transforms_HIGH)
        apply_LOW = transforms.Compose(self.transforms_LOW)

        Image_HIGH = apply_HIGH(Image.open(target_path).convert("RGB"))
        Image_LOW = apply_LOW(Image_HIGH)
        

        return Image_HIGH, Image_LOW

class image_process:
  def image_upscale(self, x_low, x_high_size):
    batch, channel, height, width = x_high_size
    up_object = transforms.Resize((height, width))
    up_image = up_object(x_low)
    return up_image

  def crop_images(self):
    pass

def compute_metrics_images(data_directory, batch_size = 16):
    dataset_faces = Dataset(data_directory)
    dataloader_faces = DataLoader(dataset_faces, batch_size = batch_size)

    norm_HIGH, norm_LOW = get_mean_std(dataloader_faces)

    print(norm_HIGH)
    print(norm_LOW)

    with open("Data\\mean_std_HIGH.npy", 'wb') as pipe:
        np.save(pipe, norm_HIGH)
    with open("Data\\mean_std_LOW.npy", 'wb') as pipe:
        np.save(pipe, norm_LOW)

    return norm_HIGH, norm_LOW

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    num_batches = 0
    channels_sum_HIGH = 0
    channels_sqrd_sum_HIGH = 0
    channels_sum_LOW = 0
    channels_sqrd_sum_LOW = 0
    for index, data in tqdm(enumerate(loader)):
        image_HIGH = convert_image_dtype(data[1])
        image_LOW = convert_image_dtype(data[0])

        channels_sum_HIGH += torch.mean(image_HIGH, axis=(0,2,3))
        channels_sqrd_sum_HIGH += torch.mean(image_HIGH ** 2, axis=(0,2,3))

        channels_sum_LOW += torch.mean(image_LOW, axis=(0,2,3))
        channels_sqrd_sum_LOW += torch.mean(image_LOW ** 2, axis=(0,2,3))
        num_batches += 1

    mean_HIGH = (channels_sum_HIGH / num_batches).numpy()
    std_HIGH = ((channels_sqrd_sum_HIGH / num_batches - mean_HIGH ** 2) ** 0.5).numpy()

    mean_LOW = (channels_sum_LOW / num_batches).numpy()
    std_LOW = ((channels_sqrd_sum_LOW / num_batches - mean_LOW ** 2) ** 0.5).numpy()

    values_HIGH = [mean_HIGH, std_HIGH]
    values_LOW = [mean_LOW, std_LOW]

    return values_HIGH, values_LOW
