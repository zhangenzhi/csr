import numpy as np
import cv2 as cv
import torch
import random
from map.quadtree import FixedQuadTree

class SegPatchify(torch.nn.Module):
    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16) -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        
    def forward(self, img, mask):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        # self.smooth_factor = 0
        if self.smooth_factor ==0 :
            edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            # edges = np.random.uniform(low=0,high=1, size=(256,256))
        else:
            grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
            edges = cv.Canny(grey_img, self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, _, _ = qdt.serialize(img, size=(self.patch_size, self.patch_size,3))
        seq_img = np.asarray(seq_img)
        seq_img = np.reshape(seq_img, [self.patch_size*self.patch_size, -1, 3])
        
        seq_mask, _, _ = qdt.serialize(mask, size=(self.patch_size, self.patch_size, 1))
        seq_mask = np.asarray(seq_mask)
        seq_mask = np.reshape(seq_mask, [self.patch_size*self.patch_size, -1, 1])

        return seq_img, seq_mask, qdt
    
class ImagePatchify(torch.nn.Module):
    def __init__(self, sths=[0,1,3,5], fixed_length=196, cannys=[50, 100], patch_size=16, num_channels=3) -> None:
        super().__init__()
        
        self.sths = sths
        self.fixed_length = fixed_length
        self.cannys = [x for x in range(cannys[0], cannys[1], 1)]
        self.patch_size = patch_size
        self.num_channels = num_channels
        
    def forward(self, img):  # we assume inputs are always structured like this
        # Do some transformations. Here, we're just passing though the input
        self.smooth_factor = random.choice(self.sths)
        c = random.choice(self.cannys)
        self.canny = [c, c+50]
        # self.smooth_factor = 0
        if self.smooth_factor ==0 :
            edges = np.random.uniform(low=0,high=1,size=(img.shape[0],img.shape[1]))
            # edges = np.random.uniform(low=0,high=1, size=(256,256))
        else:
            grey_img = cv.GaussianBlur(img, (self.smooth_factor, self.smooth_factor), 0)
            edges = cv.Canny(grey_img, self.canny[0], self.canny[1])

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)
        seq_img, seq_size, seq_pos = qdt.serialize(img, size=(self.patch_size,self.patch_size, self.num_channels))
        seq_size = np.asarray(seq_size)
        seq_img = np.asarray(seq_img)
        seq_img = np.reshape(seq_img, [self.patch_size*self.patch_size, -1, self.num_channels])

        return seq_img, seq_size, seq_pos