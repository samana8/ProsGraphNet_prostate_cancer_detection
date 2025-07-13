"""
The following code is adapted from: https://github.com/ignacio-rocco/cnngeometric_pytorch.
"""

from __future__ import print_function, division
import torch
from skimage import io
from collections import OrderedDict
from image.normalization import NormalizeImageDict, normalize_image
from geotnf.transformation import GeometricTnf
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from skimage.metrics import hausdorff_distance


def train(epoch,model,loss_fn,optimizer,dataloader,pair_generation_tnf,use_cuda=True,log_interval=50, lr=0.005, num_segments=80, model_name=None):
    if model_name is None:
        raise Exception("model name is required")
    
    model.train()
    train_loss = 0
    file_name = f'train_LR{lr}_num_seg_{num_segments}_{model_name}.log'
    log_path = './training_data/training_logs/'
    file_path = os.path.join(log_path, file_name)

    os.makedirs(log_path, exist_ok=True)

    if not os.path.isfile(file_path):
        with open(file_path, 'w+') as f:
            print(f"file {file_path} created")
    
    with open(file_path, 'a+') as f:
        for batch_idx, batch in tqdm(enumerate(dataloader), desc='training-epoch'):
            optimizer.zero_grad()
            tnf_batch = pair_generation_tnf(batch)
            if torch.isnan(tnf_batch['source_mask']).any() or torch.isnan(tnf_batch['target_mask']).any():
                print("NaNs found in input batch train")
                continue
            theta = model(tnf_batch)
            loss = loss_fn(theta,tnf_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.data.cpu().numpy()
            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({100. * batch_idx / len(dataloader)}%)]\t\tLoss: {loss.data}')
                f.write(f'Train Epoch: {epoch} [{batch_idx}/{len(dataloader)} ({100. * batch_idx / len(dataloader)}%)]\t\tLoss: {loss.data}\n')
                
        train_loss /= len(dataloader)
        print(f'Train set: Average loss: {train_loss}')
        f.write(f'Train set: Average loss: {train_loss}\n')

    return train_loss

def test(model,loss_fn,dataloader,pair_generation_tnf,use_cuda=True,geometric_model='affine', lr=0.005, num_segments=80):
    model.eval()
    test_loss = 0
    dice = 0

    file_name = f'test_LR{lr}_num_seg_{num_segments}_{geometric_model}.log'
    log_path = './training_data/training_logs/'
    file_path = os.path.join(log_path, file_name)

    os.makedirs(log_path, exist_ok=True)

    if not os.path.isfile(file_path):
        with open(file_path, 'w+') as f:
            print(f"file {file_path} created")
    
    with open(file_path, 'a+') as f:
        for batch_idx, batch in tqdm(enumerate(dataloader), desc='testing-epoch'):
            
            tnf_batch = pair_generation_tnf(batch)
            if torch.isnan(tnf_batch['source_mask']).any() or torch.isnan(tnf_batch['target_mask']).any():
                print("NaNs found in input batch test")
                continue

            theta = model(tnf_batch)
            loss = loss_fn(theta,tnf_batch)
            test_loss = loss.detach().cpu().numpy() if loss.numel() > 1 else loss.item()
            
            I = tnf_batch['target_mask']
            geometricTnf = GeometricTnf(geometric_model, 240, 240, use_cuda = use_cuda)

            if geometric_model == 'affine':
                theta = theta.view(-1,2,3)
            J = geometricTnf(tnf_batch['source_mask'],theta)
            
            if use_cuda:
                I = I.cuda(non_blocking=True)
                J = J.cuda(non_blocking=True)

            print(f'Sum of I: {torch.sum(I)}')
            print(f'Sum of J: {torch.sum(J)}')
            
            # numerator = 2 * torch.sum(torch.sum(torch.sum(I * J,dim=3),dim=2),dim=1)
            # denominator = torch.sum(torch.sum(torch.sum(I + J,dim=3),dim=2),dim=1)
            # dice = dice + torch.sum(numerator/(denominator + 0.00001))/I.shape[0]

            # Calculate Dice coefficient
            intersection = (I * J).sum(dim=[1, 2, 3])
            denominator = I.sum(dim=[1, 2, 3]) + J.sum(dim=[1, 2, 3])
            dice += (2 * intersection / (denominator + 1e-5)).mean().item()

            # Calculate Hausdorff distance for each sample in the batch
            for i in range(I.size(0)):
                I_np = I[i].cpu().numpy()
                J_np = J[i].cpu().numpy()
                hausdorff = hausdorff_distance(I_np, J_np)
                hausdorff_total += hausdorff

        test_loss /= len(dataloader)
        dice /= len(dataloader)
        hausdorff_average = hausdorff_total / (len(dataloader) * I.size(0))
    
        print(f'Test set: Average loss: {test_loss}')
        print(f'Test set: Dice: {dice}')
        print(f'Test set: Hausdorff distance: {hausdorff_average}')

        f.write(f'Test set: Average loss: {test_loss}\n')
        f.write(f'Test set: Dice: {dice}\n')
        f.write(f'Test set: Hausdorff distance: {hausdorff_average}\n')

    return test_loss
