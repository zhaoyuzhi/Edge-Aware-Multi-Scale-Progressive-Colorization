import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def get_gray(self, x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim = 1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.get_gray(x)
        x_v = F.conv2d(x, self.weight_v, padding = 1)
        x_h = F.conv2d(x, self.weight_h, padding = 1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)
        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)

class HuberLoss(nn.Module):

    def __init__(self, theta = 1):
        super(HuberLoss, self).__init__()
        self.theta = theta
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def forward(self, output, gt_img):
        diff = torch.abs(output - gt_img)
        cond = diff <= self.theta
        loss = torch.where(cond, \
            0.5 * self.l2loss(output, gt_img), \
                self.theta * self.l1loss(output, gt_img) - 0.5 * self.theta * self.theta)
        return torch.mean(loss)

def Trainer(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    
    # Loss functions
    criterion_huber = HuberLoss().cuda()
    criterion_grad = GradLoss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Save the model
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = 'EMPC_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
        if opt.save_mode == 'iter':
            model_name = 'EMPC_iter%d_bs%d.pth' % (iteration, opt.batch_size)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.ColorizationDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_L, true_RGB) in enumerate(dataloader):

            # To device
            true_L = true_L.cuda()
            true_RGB = true_RGB.cuda()
            true_RGB_mid = F.interpolate(true_RGB, scale_factor = 0.5, mode = 'nearest')
            true_RGB_bottom = F.interpolate(true_RGB, scale_factor = 0.25, mode = 'nearest')

            # Train Generator
            optimizer_G.zero_grad()
            top, mid, bottom = generator(true_L)
            
            # Pixel-level L1 Loss
            loss_huber = criterion_huber(top, true_RGB) + criterion_huber(mid, true_RGB_mid) + criterion_huber(bottom, true_RGB_bottom)
            
            # Gradient Loss
            loss_grad = criterion_grad(top, true_RGB) + criterion_grad(mid, true_RGB_mid) + criterion_grad(bottom, true_RGB_bottom)

            # Overall Loss and optimize
            loss = loss_huber + loss_grad
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Huber Loss: %.4f] [Gradient Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_huber.item(), loss_grad.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [top, true_RGB]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = opt.sample_path, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list)
