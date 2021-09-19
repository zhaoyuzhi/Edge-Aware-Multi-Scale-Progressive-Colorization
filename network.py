import torch
import torch.nn as nn
import torch.nn.functional as F

from network_module import *

# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# ----------------------------------------
#                Generator
# ----------------------------------------
class EMPC_UNet(nn.Module):
    def __init__(self, opt, in_channels):
        super(EMPC_UNet, self).__init__()
        # Downsample blocks
        self.down1_1 = Conv2dLayer(in_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = 'none')
        self.down1_2 = Conv2dLayer(opt.start_channels, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.down2_1 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.down2_2 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.down3_1 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.down3_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.down3_3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Bottleneck
        self.b1_1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b1_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b1_3 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b2_1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b2_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b2_3 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b3_1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b3_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b3_3 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 2, 2, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b4_1 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b4_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.b4_3 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        # Fusion & Upsample
        self.up1_1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up1_2 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up1_3 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up2_1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up2_2 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up3_1 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ, norm = opt.norm)
        self.up3_2 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 3, 1, 1, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
    
    def forward(self, x):
        '''
        # Pack Grayscale Input
        if x.shape[1] == 1:
            x = torch.cat((x, x, x), 1)
        '''
        # Encoder
        down1 = self.down1_1(x)                                     # out: batch * 64 * H * W
        down1 = self.down1_2(down1)                                 # out: batch * 64 * H * W
        down2 = self.down2_1(down1)                                 # out: batch * 128 * (H/2) * (W/2)
        down2 = self.down2_2(down2)                                 # out: batch * 128 * (H/2) * (W/2)
        down3 = self.down3_1(down2)                                 # out: batch * 256 * (H/4) * (W/4)
        down3 = self.down3_2(down3)                                 # out: batch * 256 * (H/4) * (W/4)
        down3 = self.down3_3(down3)                                 # out: batch * 256 * (H/4) * (W/4)
        # Bottleneck
        b = self.b1_1(down3)                                        # out: batch * 512 * (H/8) * (W/8)
        b = self.b1_2(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b1_3(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b2_1(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b2_2(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b2_3(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b3_1(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b3_2(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b3_3(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b4_1(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b4_2(b)                                            # out: batch * 512 * (H/8) * (W/8)
        b = self.b4_3(b)                                            # out: batch * 512 * (H/8) * (W/8)
        # Decoder
        up1 = self.up1_1(b)                                         # out: batch * 256 * (H/4) * (W/4)
        up1 = torch.cat((up1, down3), 1)                            # out: batch * 512 * (H/4) * (W/4)
        up1 = self.up1_2(up1)                                       # out: batch * 256 * (H/4) * (W/4)
        up1 = self.up1_3(up1)                                       # out: batch * 256 * (H/4) * (W/4)
        up2 = self.up2_1(up1)                                       # out: batch * 128 * (H/2) * (W/2)
        up2 = torch.cat((up2, down2), 1)                            # out: batch * 256 * (H/2) * (W/2)
        up2 = self.up2_2(up2)                                       # out: batch * 128 * (H/2) * (W/2)
        up3 = self.up3_1(up2)                                       # out: batch * 64 * H * W
        up3 = torch.cat((up3, down1), 1)                            # out: batch * 128 * H * W
        up3 = self.up3_2(up3)                                       # out: batch * 3 * H * W
        return up3

class EMPC(nn.Module):
    def __init__(self, opt):
        super(EMPC, self).__init__()
        self.bottom = EMPC_UNet(opt, opt.in_channels)
        self.mid = EMPC_UNet(opt, opt.in_channels + opt.out_channels)
        self.top = EMPC_UNet(opt, opt.in_channels + opt.out_channels)

    def forward(self, x):
        # Bottom
        x_bottom = F.interpolate(x, scale_factor = 0.25, mode = 'nearest')
        bottom = self.bottom(x_bottom)
        # Mid
        x_mid = F.interpolate(x, scale_factor = 0.5, mode = 'nearest')
        bottom_ = F.interpolate(bottom, scale_factor = 2, mode = 'nearest')
        x_mid = torch.cat((x_mid, bottom_), 1)
        mid = self.mid(x_mid)
        # Top
        mid_ = F.interpolate(mid, scale_factor = 2, mode = 'nearest')
        x = torch.cat((x, mid_), 1)
        top = self.top(x)
        return top, mid, bottom

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv3_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channel for U-Net decoder')
    parser.add_argument('--pad', type = str, default = 'zero', help = 'padding type')
    parser.add_argument('--activ', type = str, default = 'relu', help = 'activation function for generator')
    parser.add_argument('--norm', type = str, default = 'bn', help = 'normalization type for generator')
    opt = parser.parse_args()

    net = EMPC_UNet(opt, 1).cuda()
    a = torch.randn(1, 1, 64, 64).cuda()
    b = net(a)
    print(b.shape)

    net = EMPC(opt).cuda()
    a = torch.randn(1, 1, 256, 256).cuda()
    top, mid, bottom = net(a)
    print(top.shape, mid.shape, bottom.shape)
