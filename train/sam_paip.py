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
from dataset.paip import PAIPAP

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
    # Create an instance of the U-Net model and other necessary components
    patch_size=args.patch_size
    sqrt_len=int(math.sqrt(args.fixed_length))
    num_class = 2 
    
    model = SAMQDT(image_shape=(patch_size*sqrt_len, patch_size*sqrt_len),
            patch_size=args.patch_size,
            output_dim=num_class, 
            pretrain=args.pretrain,
            qdt=True)
    criterion = DiceBLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    best_val_score = 0.0
    
    # Move the model to GPU
    model = nn.DataParallel(model)
    model.to(device)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    # Define the learning rate scheduler
    milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    data_path = args.data_dir
    dataset = PAIPAP(data_path, args.resolution, fixed_length=args.fixed_length, patch_size=patch_size, normalize=False)
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    test_size = val_size
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, dataset_size))
    train_set = Subset(dataset, train_indices)
    val_set = test_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Training loop
    num_epochs = args.epoch
    train_losses = []
    val_losses = []
    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, qimages, _, qmasks, _ = batch
            qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
        
            outputs = model(qimages)
            loss = criterion(outputs, qmasks)
                
            # print("train step loss:{}".format(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_train_loss += loss.item()
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}".format(end_time-start_time,(end_time-start_time)/train_size))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_score = 0.0
        epoch_qdt_score = 0.0
        epoch_qmask_score = 0.0
        with torch.no_grad():
            for bi,batch in enumerate(val_loader):
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                image, qimages, mask, qmasks, qdt_info = batch
                seq_shape = qmasks.shape
                qimages = torch.reshape(qimages,shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
                qmasks = torch.reshape(qmasks,shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
                qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
                outputs = model(qimages)
                loss = criterion(outputs, qmasks)
                score = dice_score(outputs, qmasks)
                # if  (epoch - 1) % 10 == 9:  # Adjust the frequency of visualization
                #     outputs = torch.reshape(outputs, seq_shape)
                #     qmasks = torch.reshape(qmasks, seq_shape)
                    # qdt_score, qmask_score = sub_trans_plot(image, mask, qmasks=qmasks, pred_mask=outputs, qdt_info=qdt_info, 
                    #                            fixed_length=args.fixed_length, bi=bi, epoch=epoch, output_dir=args.savefile)
                    # epoch_qdt_score += qdt_score.item()
                    # epoch_qmask_score += qmask_score.item()
                epoch_val_loss += loss.item()
                epoch_val_score += score.item()

        epoch_val_loss /= len(val_loader)
        epoch_val_score /= len(val_loader)
        epoch_qdt_score /= len(val_loader)
        epoch_qmask_score /= len(val_loader)
        val_losses.append(epoch_val_loss)
        # Save the best model based on validation accuracy
        if epoch_val_score > best_val_score:
            best_val_score = epoch_val_score
            torch.save(model.state_dict(), os.path.join(args.savefile, "best_score_model.pth"))
            logging.info(f"Model save with dice score {best_val_score} at epoch {epoch}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f},\
            Score: {epoch_val_score:.4f} QDT Score: {epoch_qdt_score:.4f}/{epoch_qmask_score:.4f}.")

    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    # Test the model
    model.eval()
    test_loss = 0.0
    epoch_test_score = 0
    with torch.no_grad():
        for batch in test_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            _, qimages, _, qmasks, _, qdt_value = batch
            qimages = torch.reshape(qimages, shape=(-1,3,patch_size*sqrt_len, patch_size*sqrt_len))
            qmasks = torch.reshape(qmasks, shape=(-1,num_class,patch_size*sqrt_len, patch_size*sqrt_len))
            qimages, qmasks = qimages.to(device), qmasks.to(device)  # Move data to GPU
            outputs = model(qimages)
            loss = criterion(outputs, qmasks)
            score = dice_score(outputs, qmasks)
            test_loss += loss.item()
            epoch_test_score += score.item()

    test_loss /= len(test_loader)
    epoch_test_score /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Score: {epoch_test_score:.4f}")

def sub_trans_plot(image, mask, qmasks, pred_mask, qdt_info, fixed_length, bi, epoch, output_dir):
    true_score = 0 
    best_score = 0
    for i in range(image.size(0)):
        image = image[i].cpu().permute(1, 2, 0).numpy()
        mask_true = mask[i].cpu().numpy()

        qmasks = (qmasks[i].cpu() > 0.5).numpy()
        qmasks.astype(np.int32)
        qmasks = qmasks[1]
        patch_size = qmasks.shape[0]
        qmasks = np.reshape(qmasks, (fixed_length, patch_size, patch_size))
        qmasks = np.repeat(np.expand_dims(qmasks, axis=-1), 3, axis=-1)
        
        # qmasks = (qmasks[i].cpu() > 0.5).numpy()
        # qmasks.astype(np.int32)
        
 
        # Squeeze the singleton dimension from mask_true
        mask_true = mask_true[1]
        mask_true = np.repeat(np.expand_dims(mask_true, axis=-1), 3, axis=-1)
        
        # print(mask_true.sum())
        pred_mask = (pred_mask[i].cpu() > 0.5).numpy()
        mask_pred = pred_mask[1]
        patch_size = mask_pred.shape[0]
        mask_pred = np.reshape(mask_pred, (fixed_length, patch_size, patch_size))
        mask_pred = np.repeat(np.expand_dims(mask_pred, axis=-1), 3, axis=-1)
      
        meta_info = []
        for nodes in qdt_info:
            n = []
            for idx in range(len(nodes)):
                n.append(nodes[idx][i].numpy())
            meta_info.append(n)
        
        qdt = FixedQuadTree(domain=mask_true, fixed_length=fixed_length, build_from_info=True, meta_info=meta_info)
        deoced_mask_pred = qdt.deserialize(seq=mask_pred, patch_size=patch_size, channel=3)
        decode_qmask = qdt.deserialize(seq=qmasks, patch_size=patch_size, channel=3)
        
        true_score += dice_score_plot(mask_true, targets=deoced_mask_pred)
        best_score += dice_score_plot(mask_true, targets=decode_qmask)
        
        mask_true = mask_true.astype(np.float64)

        # Plot and save images
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_true, cmap='gray')
        plt.title("True Mask")

        plt.subplot(1, 3, 3)
        plt.imshow(deoced_mask_pred, cmap='gray')
        plt.title("Predicted Mask")
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch + 1}_sample_{bi + 1}.png"))
        plt.close()
        # true_score /= image.size(0)
        return true_score, best_score

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
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./sam-paip-ap",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    main(args)
