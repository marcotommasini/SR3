from locale import normalize
import os
import torch
from torchvision.io import read_image
from torch.utils import data
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms as transforms

import cv2
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import zipfile


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for index, data in enumerate(loader):
        image_HIGH = data[1]
        image_LOW = data[0]

        channels_sum_HIGH += torch.mean(image_HIGH, dim=[0, 2, 3])
        channels_sqrd_sum_HIGH += torch.mean(image_HIGH ** 2, dim=[0, 2, 3])

        channels_sum_LOW += torch.mean(image_LOW, dim=[0, 2, 3])
        channels_sqrd_sum_LOW += torch.mean(image_LOW ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean_HIGH = channels_sum_HIGH / num_batches
    std_HIGH = (channels_sqrd_sum_HIGH / num_batches - mean_HIGH ** 2) ** 0.5

    mean_LOW = channels_sum_LOW / num_batches
    std_LOW = (channels_sqrd_sum_LOW / num_batches - mean_LOW ** 2) ** 0.5

    values_HIGH = [mean_HIGH, std_HIGH]
    values_LOW = [mean_LOW, std_LOW]

    return values_HIGH, values_LOW

def unzip_file(input_directory, output_directory):
    with zipfile.ZipFile(input_directory, 'r') as zip_ref:
        zip_ref.extractall(output_directory)

class DataSet_Faces(data.Dataset):
    def __init__(self, dataset_directory = None, image_dimensions = (128,128), downsample_dimensions = (16,16), norm_HIGH = None, norm_LOW = None):
        super().__init__()
        self.dataset_directory = dataset_directory
        self.image_dimensions = image_dimensions
        self.downsample_dimensions = downsample_dimensions
        self.list_images = [image_name for image_name in open(self.dataset_directory)]
        self.norm_HIGH = norm_HIGH
        self.norm_LOW = norm_LOW

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_directory, self.list_images[index])

        image_HIGH = np.asarray(Image.open(img_path).convert("RGB"))
        image_LOW = image_HIGH.resize( self.downsample_dimensions, Image.ANTIALIAS)
        
        
        if self.norm_HIGH == None and self.norm_LOW == None:
            transform_LOW = transforms.Compose([transforms.ToTensor(),\
                                                transforms.normalize(self.norm_HIGH[0], self.norm_HIGH[1])])
            transform_HIGH = transforms.Compose([transforms.ToTensor(),\
                                            transforms.normalize(self.norm_LOW[0], self.norm_LOW[1])])
        else:
            transform_LOW = transforms.Compose([transforms.ToTensor()])
                                               
            transform_HIGH = transforms.Compose([transforms.ToTensor()])
                                            

        image_HIGH_tr = transform_LOW(image_HIGH)
        image_LOW_tr = transform_HIGH(image_LOW)


        return image_HIGH_tr, image_LOW_tr
        
        