import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # general parameters
    parser.add_argument('--load_name', type = str, \
        default = './models/EMPC_epoch40_bs64.pth', \
            help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of cpu threads to use during batch generation')
    parser.add_argument('--save_path', type = str, default = 'EMPC_val_results', help = 'number of cpu threads to use during batch generation')
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channel for U-Net decoder')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'padding type')
    parser.add_argument('--activ', type = str, default = 'relu', help = 'activation function for generator')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type for generator')
    # dataset
    parser.add_argument('--baseroot', type = str, \
        default = '/home/mybeast/zhaoyuzhi/ILSVRC2012_val_256', \
            help = 'color image baseroot')
    parser.add_argument('--vallist', type = str, \
        default = 'ctest10k.txt', \
            help = 'vallist')
    parser.add_argument('--crop_size', type = int, default = 256, help = 'single patch size')
    opt = parser.parse_args()
    print(opt)

    # Create saving folder
    utils.check_path(opt.save_path)
    
    # Define the network
    generator = utils.create_generator(opt).cuda()
    generator.eval()

    # Define the dataset
    trainset = dataset.ColorizationDataset_Val(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # For loop training
    for i, (true_L, true_RGB, imgname) in enumerate(dataloader):

        # print
        imgname = imgname[0]
        savename = os.path.join(opt.save_path, imgname)
        print(i, savename)

        # To device
        true_L = true_L.cuda()
        true_RGB = true_RGB.cuda()

        # Forward
        with torch.no_grad():
            top, mid, bottom = generator(true_L)

        # Recover normalization: * 255 because last layer is sigmoid activated
        top = top * 255.0
        # Process img_copy and do not destroy the data of img
        img_copy = top.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, 255)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        # Save validation images
        img_copy = cv2.resize(img_copy, (opt.crop_size, opt.crop_size))
        cv2.imwrite(savename, img_copy)
