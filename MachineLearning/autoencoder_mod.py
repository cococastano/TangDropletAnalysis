import torch
import torchvision
import matplotlib.pyplot as plt
import data_utils
import numpy as np
import torch.nn.functional as F
import os
import cv2
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

"""
Nicolas Castano
Last revision: 6/17/18

Module to import for using trained autoencoder models from conv_autoencoder.
Class has methods for reducing to fully connected layer and viewing encoded images.

"""


############################## CLASSES AND METHODS ############################
def flatten(x):
    # read in N, C, H, W
    N = x.shape[0]
    # flatten the the C * H * W images into a single vector per image
    return x.view(N, -1)
  
def to_img(x):
#    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    try:
        x = x.view(x.size(0), 1, x.shape[2], x.shape[3])
    except:
        x = x.view(x.shape[0], x.shape[1])
    return x


class autoencoder(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, num_dims=3):
        super(autoencoder, self).__init__()
        # encode
        self.conv_2d_1 = nn.Conv2d(in_channel, channel_1, kernel_size=7, 
                                   stride=1, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2d_2 = nn.Conv2d(channel_1, channel_2,kernel_size=3, 
                                   stride=1, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_2d_3 = nn.Conv2d(channel_2, channel_3, stride=1,
                                   kernel_size=3, padding=1)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # decode   
        self.conv_trans_2d_1 = nn.ConvTranspose2d(channel_3, channel_2, 
                                                  kernel_size=4, stride=2,
                                                  padding=1)
        self.conv_trans_2d_2 = nn.ConvTranspose2d(channel_2, channel_1, 
                                                  kernel_size=2, stride=2)
        self.conv_trans_2d_3 = nn.ConvTranspose2d(channel_1, in_channel, 
                                                  kernel_size=4, stride=2, 
                                                  padding=1)
        self.maxpool_decode_2 = nn.MaxPool2d(kernel_size=3, stride=1,
                                             padding=1)

        # encode to n dims
        self.num_dims = num_dims  
        # know last channel for fully connected layer
        self.channel_3 = channel_3
        
    def decode(self, x):
        x = F.relu(self.conv_trans_2d_1(x))
        x = self.maxpool_decode_2(x)
        x = F.relu(self.conv_trans_2d_2(x))
        x = F.relu(self.conv_trans_2d_3(x))
        return x
    
    def encode(self, x):
        x = F.relu(self.conv_2d_1(x))
        x = self.maxpool_1(x)
        x = F.relu(self.conv_2d_2(x))
        x = self.maxpool_2(x)
        x = F.relu(self.conv_2d_3(x))
        x = self.maxpool_3(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def get_encoded_im(self,x,im_i=None,layer=None):
        x = self.encode(x)
        x = x.detach().numpy()
        if im_i is None and layer is None: 
            out = x[:,:,:,:]
        elif im_i is None and layer is not None:
            out = x[:,layer,:,:]
        elif im_i is not None and layer is None:
            out = x[im_i,:,:,:]
        elif im_i is not None and layer is not None:
            out = x[im_i,layer,:,:]
        return out
    
    def get_decoded_im(self,x,im_i=0,layer=0):
        x = self.encode(x)
        x = self.decode(x)
        x = x.detach().numpy()
        out = x[im_i,layer,:,:]
        return out
    
    def encode_to_n_dims(self,x,n):
        x = self.encode(x)
        fc_1_encode = nn.Linear(x.shape[2]*x.shape[3]*self.channel_3,500)
        fc_2_encode = nn.Linear(500,100)
        fc_3_encode = nn.Linear(100,self.num_dims)
        x = flatten(x)
        x = fc_1_encode(x)
        x = fc_2_encode(x)
        if n is self.num_dims:
            x = fc_3_encode(x)
        else:
            fc_last = nn.Linear(100, n)
            x = fc_last(x)
        return x


