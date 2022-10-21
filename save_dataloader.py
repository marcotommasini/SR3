from Data.dataset import DataSet_Faces, get_mean_std
from utils import get_statistical_parameters
from torch.utils.data import DataLoader
import torch
from dataset_transform import TransformDataset
import torchvision.transforms as transforms
import numpy as np
import sys

def save_dataloader():
    #This is only valid for this specific dataset
    mean_HIGH = [121.44110452, 113.7276543 , 105.44456356]
    std_HIGH = [75.26942776, 74.99163981, 77.8582931 ]

    mean_LOW = [132.66010058, 108.41433053,  96.97006887]
    std_LOW = [71.57009842, 64.93825117, 65.14655061]


    transform_HIGH  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean_HIGH), (std_HIGH))])
    transform_LOW = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean_LOW), (std_LOW))])
    dataset_directory =  "Data\\thumbnails128x128"

    dataset = DataSet_Faces(dataset_directory)

    trsdataset = TransformDataset(dataset,(transform_HIGH, transform_LOW))

    dataloader = DataLoader(trsdataset, 16, num_workers=4, drop_last=True)

    path = "Data\\dataloader.pth"
    print("Saved Dataloader")

    torch.save(dataloader,path)

def main():
    loader = torch.load("Data\\dataloader.pth")

    print(loader.__len__())

if __name__ == '__main__':
    main()
