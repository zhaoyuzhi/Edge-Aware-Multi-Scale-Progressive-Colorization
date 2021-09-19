import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import utils

class ColorizationDataset(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_jpgs(opt.baseroot)
        '''
        if opt.smaller_coeff > 1:
            imglist = self.create_sub_trainset(imglist, opt.smaller_coeff)
        '''
        self.imglist = imglist

    def create_sub_trainset(self, imglist, smaller_coeff):
        # Sample the target images
        namelist = []
        for i in range(len(imglist)):
            if i % smaller_coeff == 0:
                a = random.randint(0, smaller_coeff - 1) + i
                namelist.append(imglist[a])
        return namelist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot, imgname)

        # Read the image
        img = cv2.imread(imgpath)

        # Process the image
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # Grayscale input image

        # Normalized to [0, 1] and to PyTorch Tensor
        grayimg = torch.from_numpy(grayimg / 255.0).float().unsqueeze(0).contiguous()
        img = torch.from_numpy(img / 255.0).float().permute(2, 0, 1).contiguous()
        
        return grayimg, img
    
    def __len__(self):
        return len(self.imglist)

class ColorizationDataset_Val(Dataset):
    def __init__(self, opt):
        # Note that:
        # 1. opt: all the options
        # 2. imglist: all the image names under "baseroot"
        self.opt = opt
        imglist = utils.get_files(opt.baseroot)
        self.imglist = imglist

    def __getitem__(self, index):
        # Path of one image
        imgname = self.imglist[index]
        imgpath = os.path.join(self.opt.baseroot, imgname)

        # Read the images
        img = cv2.imread(imgpath)
        h, w = img.shape[0], img.shape[1]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # RGB output image
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # Grayscale input image

        # Normalized to [0, 1] and to PyTorch Tensor
        grayimg = torch.from_numpy(grayimg / 255.0).float().unsqueeze(0).contiguous()
        img = torch.from_numpy(img / 255.0).float().permute(2, 0, 1).contiguous()
        
        return grayimg, img, imgpath, h, w
    
    def __len__(self):
        return len(self.imglist)

if __name__ == "__main__":
    
    a = torch.randn(1, 3, 256, 256)
    b = a[:, [0], :, :] * 0.299 + a[:, [1], :, :] * 0.587 + a[:, [2], :, :] * 0.114
    b = torch.cat((b, b, b), 1)
    print(b.shape)

    c = torch.randn(1, 1, 256, 256)
    d = a * c
    print(d.shape)
