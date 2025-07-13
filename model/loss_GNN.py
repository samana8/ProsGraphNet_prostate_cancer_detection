from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf
from geotnf.transformation import GeometricTnf
from skimage import io

class SSDLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True):
        super(SSDLoss, self).__init__()
        self.geometric_model = geometric_model
        self.use_cuda = use_cuda
        self.geometricTnf = GeometricTnf(self.geometric_model, 240, 240, use_cuda=self.use_cuda)

    def forward(self, theta, tnf_batch):
        ### compute square root of SSD
        A = tnf_batch['target_image']
        B = self.geometricTnf(tnf_batch['source_image'], theta)
        
        ssd = torch.sum((A - B) ** 2, dim=[1, 2, 3])
        ssd = torch.sqrt(ssd / (A.shape[0] * A.shape[1] * A.shape[2] * A.shape[3]))
        
        return ssd.mean()
