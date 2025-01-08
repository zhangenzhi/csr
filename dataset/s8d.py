import os
import sys
sys.path.append("./")
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from map.transform import ImagePatchify
import cv2 as cv

# Set the flag to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class S8D(Dataset):
    def __init__(self, data_dir, resolution):
        self.data_dir = data_dir
        self.resolution = resolution
        self.image_filenames = []

        for i in range(len(os.listdir(data_dir))):
            sample_name = f"volume_{str(i).zfill(3)}.raw"
            image_path = os.path.join(data_dir, sample_name)
            if os.path.exists(image_path):
                self.image_filenames.extend([image_path])
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        file = self.image_filenames[idx]
        image = np.fromfile(file, dtype=np.uint16).reshape([self.resolution, self.resolution])
        edge = cv.Canny(cv.GaussianBlur((image[:] / 255), (5,5), 0).astype(np.uint8), 50, 90)
        edge = edge.astype(np.float32)
        img = (image/255).astype(np.float32)
        return img, edge
    
class S8DAP(Dataset):
    def __init__(self, data_dir, resolution, fixed_length=1024, sths=[0,1,3,5,7], cannys=[50, 100], patch_size=16, ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.image_filenames = []
        self.patchify = ImagePatchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size, num_channels=1)

        self.transform =  transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.seq_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        for i in range(len(os.listdir(data_dir))):
            sample_name = f"volume_{str(i).zfill(3)}.raw"
            image_path = os.path.join(data_dir, sample_name)
            if os.path.exists(image_path):
                self.image_filenames.extend([image_path])
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        
        file = self.image_filenames[idx]
        image = np.fromfile(file, dtype=np.uint16).reshape([self.resolution, self.resolution, 1])
        image = (image[:] / 255).astype(np.uint8)
        seq_img, seq_size, _ = self.patchify(image)
        
        seq_img = self.seq_transform(seq_img)
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, seq_img

    
if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="paip", 
                        help='base path of dataset.')
    parser.add_argument('--datapath', default="/Volumes/data/dataset/s8d/", 
                        help='base path of dataset.')
    parser.add_argument('--resolution', default=512, type=int,
                        help='resolution of img.')
    parser.add_argument('--fixed_length', default=512, type=int,
                        help='length of sequence.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./vitunet_visual",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    data_dir="/Volumes/data/dataset/s8d/"
    resolution=8192
    # dataset = S8D(data_dir, resolution)
    dataset = S8DAP(data_dir, resolution, fixed_length=1024)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Now you can iterate over the dataloader to get batches of images and masks
    for batch in dataloader:
        # image, qdt_img, mask, qdt_mask, qdt_info = batch
        # print(image.shape, qdt_img.shape, mask.shape, qdt_mask.shape)
        
        image, qdt_img = batch
        print(image.shape, qdt_img.shape)