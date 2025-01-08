import os
import sys
sys.path.append("./")
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

from map.quadtree import FixedQuadTree
from model.sam import SAMQDT
from dataset.s8d import S8DAP

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class DiceBLoss(nn.Module):
    def __init__(self, weight=0.5, num_class=2, size_average=True):
        super(DiceBLoss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, inputs, targets, smooth=1, act=True):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        if act:
            inputs = F.sigmoid(inputs)       
        
        # pred = torch.flatten(inputs)
        # true = torch.flatten(targets)
        
        # #flatten label and prediction tensors
        pred = torch.flatten(inputs[:,1:,:,:])
        true = torch.flatten(targets[:,1:,:,:])
        
        intersection = (pred * true).sum()
        coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)                                        
        dice_loss = 1 - (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)  
        BCE = F.binary_cross_entropy(pred, true, reduction='mean')
        dice_bce = self.weight*BCE + (1-self.weight)*dice_loss
        # dice_bce = dice_loss 
        
        return dice_bce
    
def dice_score(inputs, targets, smooth=1):
    inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    pred = torch.flatten(inputs[:,1:,:,:])
    true = torch.flatten(targets[:,1:,:,:])
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   
    return coeff  

def dice_score_plot(inputs, targets, smooth=1):     
    #flatten label and prediction tensors
    pred = inputs[...,0].flatten()
    true = targets[...,0].flatten()
    
    intersection = (pred * true).sum()
    coeff = (2.*intersection + smooth)/(pred.sum() + true.sum() + smooth)   
    return coeff  

import logging

# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def main(args):
    
    log(args=args)
    patch_size=args.patch_size
    sqrt_len=int(math.sqrt(args.fixed_length))
    num_class = 2 
    
    model = SAMQDT(image_shape=(patch_size*sqrt_len, patch_size*sqrt_len),
            patch_size=args.patch_size,
            output_dim=num_class, 
            pretrain=args.pretrain,
            qdt=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")    
    # Move the model to GPU
    model.to(device)

    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = S8DAP(data_path, args.resolution, fixed_length=args.fixed_length, patch_size=patch_size)
    dataset_size = len(dataset)
    print("dataset size:", dataset_size)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    # Training loop
    num_epochs = args.epoch

    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            image, seq_img = batch
            seq_img = torch.reshape(seq_img,shape=(-1,1,patch_size*sqrt_len, patch_size*sqrt_len))
            seq_img = seq_img.to(device)  # Move data to GPU
        
            outputs = model(seq_img)
            print(outputs.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="../dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=8192, type=int,
                        help='resolution of img.')
    parser.add_argument('--fixed_length', default=1024, type=int,
                        help='length of sequence.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='patch size.')
    parser.add_argument('--pretrain', default="sam-b", type=str,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use SAM pretrained weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./sam-s8d-ap",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
