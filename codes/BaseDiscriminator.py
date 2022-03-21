import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from .temporal_shift import TemporalShift


class BaseDiscriminator(nn.Module):


    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                temporal_shift=False, n_segment=8, n_div=8):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BaseDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1
        
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding)
        self.act = nn.LeakyReLU(0.2)

        nf_mult = 1
        nf_mult_prev = 1
        conv_layers = []
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv_layers += [
                DownConvBlock(in_dim = ndf * nf_mult_prev,
                             out_dim = ndf * nf_mult,
                              norm_layer = norm_layer,
                             kernel_size = kernel_size,
                             stride = 2,
                             padding = padding,
                             bias = use_bias)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv_layers += [
                DownConvBlock(in_dim = ndf * nf_mult_prev,
                             out_dim = ndf * nf_mult,
                              norm_layer = norm_layer,
                             kernel_size = kernel_size,
                             stride = 1,
                             padding = padding,
                             bias = use_bias)]

        self.conv_layers = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_act = nn.Identity()
        self.classifier = nn.Linear(ndf * nf_mult, output_nc, bias=use_bias)
        
        if temporal_shift:
            self.conv_layers = self.make_block_temporal(self.conv_layers, n_segment, n_div)
        
        
    def make_block_temporal(self, stage, n_segment, n_div):
        n_round = 1
        blocks = list(stage.children())
        print('=> Processing stage with {} blocks residual'.format(len(blocks)))
        for i, b in enumerate(blocks):
            blocks[i].conv = TemporalShift(b.conv, n_segment=n_segment, n_div=n_div)
        return nn.Sequential(*blocks)
    
    
    def forward(self, x):
        """Standard forward."""
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv_layers(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        x = self.out_act(x)
        return x

    

class DownConvBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_dim, out_dim, norm_layer, kernel_size=3, stride=2, padding=1, bias=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(DownConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.act = nn.LeakyReLU(0.2)
        self.norm = norm_layer(out_dim)
        
    def forward(self, x):
        """Forward function (with skip connections)"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x       
