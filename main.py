from multiprocessing import get_start_method
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from model import UNET_SR3
from Data.dataset import get_mean_std, unzip_file, DataSet_Faces
from torch.utils.data import DataLoader
from utils import get_statistical_parameters
from functions import operations as op

def main():
    parser = argparse.ArgumentParser(description='Diffusion model')

    parser.add_argument('--device', type=str, default="cuda", help='Device to run the code on')
    parser.add_argument('--use_checkpoints', type=str, default="False", help='Use checkpoints')
    parser.add_argument('--emb_dimension', type=int, default=256, help='Number of embeded time dimension')
    parser.add_argument('--number_noise_steps', type=int, default=1000, help='Numbe of steps required to noise the image')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='First value of beta')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Last value of beta')
    parser.add_argument('--beta_curve', type=str, default="linear", help='How the value of beta will change over time')
    parser.add_argument('--target_image_size', type=int, default=128, help='Size of the squared input image')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--number_workers', type=int, default=2, help='Number of workers for the dataloader')
    parser.add_argument('--number_steps', type=int, default=200, help='How many iterations steps the model will learn from')
    parser.add_argument('--number_epochs', type=int, default=50, help='Number of epochs the model will learn from')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-6, help='Initial learning rate of the optmizer')
    parser.add_argument('--final_learning_rate', type=float, default=1e-4, help='Initial learning rate of the optmizer')
    parser.add_argument('CD','--checkpoint_directory', type=str, default="", help='Input Checkpoints directory')
    parser.add_argument('DD', '--dataset_directory', type=str, default="", help='FIle with images')
    parser.add_argument('ZDD', '--zipped_dataset_directory', type=str, default="", help='Zipped file to be unzipped')

    args = parser.parse_args()

    #UNZIP the file and get statistical parameters
    stats_HIGH, stats_LOW = get_statistical_parameters(args.zipped_dataset_directory, args.dataset_directory,batch_size=args.batch_size)

    #Load data and create datalaoder
    dataset = DataSet_Faces(args.dataset_directory, norm_HIGH = stats_HIGH, norm_LOW = stats_LOW)
    dataloader = DataLoader(dataset, args.batch_size, num_workers=4, drop_last=True)

    model = UNET_SR3()

    optmizer = torch.optim.Adam(model.parameters(), lr=args.initial_learning_rate)

    loss = nn.MSELoss()

    op_object = op(args)

    op_object.train_model(model, dataloader, optmizer, loss)

if __name__ == "__main__":
    main()



    
