from locale import normalize
import os
import torch
from torchvision.io import read_image
from torch.utils import data
from torchvision.transforms.functional import convert_image_dtype

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import zipfile
from functools import partial
from tqdm import tqdm



def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    num_batches = 0
    channels_sum_HIGH = 0
    channels_sqrd_sum_HIGH = 0
    channels_sum_LOW = 0
    channels_sqrd_sum_LOW = 0
    for index, data in tqdm(enumerate(loader)):
        image_HIGH = data[1]
        image_LOW = data[0]

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

def unzip_file(input_directory, output_directory):
    with zipfile.ZipFile(input_directory, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

class DataSet_Faces(data.Dataset):
    def __init__(self, dataset_directory = None, image_dimensions = (128,128), downsample_dimensions = (16,16)):
        super().__init__()
        self.dataset_directory = dataset_directory
        self.image_dimensions = image_dimensions
        self.downsample_dimensions = downsample_dimensions
        self.list_images = os.listdir(dataset_directory)


    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_directory, self.list_images[index])

        image_HIGH = np.asarray(Image.open(img_path).convert("RGB"), dtype=float)

        ys, xs = self.downsample_dimensions

        _, _, c  = image_HIGH.shape

        image_LOW = np.resize(image_HIGH,(ys,xs,c))

        return image_HIGH, image_LOW
        
class image_process:
  def image_upscale(self, x_low, x_high_size):
    output_size = x_high_size
    up_object = torch.nn.Upsample(size = output_size, mode= "bilinear")
    up_image = up_object(x_low)
    return up_image

  def crop_images(self):
    pass

def compute_metrics_images(data_directory, batch_size = 16):
    dataset_faces = DataSet_Faces(data_directory)
    dataloader_faces = DataLoader(dataset_faces, batch_size = batch_size)

    norm_HIGH, norm_LOW = get_mean_std(dataloader_faces)

    print(norm_HIGH)
    print(norm_LOW)

    with open("Data\\mean_std_HIGH.npy", 'wb') as pipe:
        np.save(pipe, norm_HIGH)
    with open("Data\\mean_std_LOW.npy", 'wb') as pipe:
        np.save(pipe, norm_LOW)

    return norm_HIGH, norm_LOW