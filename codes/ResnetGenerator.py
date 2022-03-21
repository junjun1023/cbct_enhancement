import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import functools
from .temporal_shift import TemporalShift

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, 
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 n_blocks=6, padding_type='reflect', n_downsampling=2,
                temporal_shift=False, n_segment=8, n_div=8):
        
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pad = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
        self.norm1 = norm_layer(ngf)
        self.act = nn.ReLU()

        down_layers = []
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down_layers += [DownConvBlock(in_dim=ngf * mult, out_dim=ngf * mult * 2, norm_layer=norm_layer, kernel_size=3, stride=2, padding=1, bias=use_bias)]
        self.down_layers = nn.Sequential(*down_layers)

        mult = 2 ** n_downsampling
        conv_layers = []
        for i in range(n_blocks):       # add ResNet blocks
            conv_layers += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, bias=use_bias)]
        self.conv_layers = nn.Sequential(*conv_layers)
        
        
        ### Upsampling
        up_layers = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            up_layers += [UpConvBlock(in_dim=ngf * mult, 
                                                                               out_dim=int(ngf * mult / 2), 
                                                                               norm_layer=norm_layer,
                                                                               kernel_size=3, stride=2, padding=1, 
                                                                               output_padding=1, bias=use_bias)]
        self.up_layers = nn.Sequential(*up_layers)

        self.conv2 = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
        self.out_act = nn.Sigmoid()
        
        
        if temporal_shift:
            self.conv_layers = self.make_block_temporal(self.conv_layers, n_segment, n_div)
            

    def make_block_temporal(self, stage, n_segment, n_div):
        n_round = 1
        blocks = list(stage.children())
        print('=> Processing stage with {} blocks residual'.format(len(blocks)))
        for i, b in enumerate(blocks):
            if i % n_round == 0:
                blocks[i].conv1 = TemporalShift(b.conv1, n_segment=n_segment, n_div=n_div)
        return nn.Sequential(*blocks)
            
            
    def forward(self, x):
        """Standard forward"""
        x = self.pad(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.down_layers(x)
        x = self.conv_layers(x)
        x = self.up_layers(x)
        x = self.pad(x)
        x = self.conv2(x)
        x = self.out_act(x)
        
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()

        self.pad = nn.Identity()
        p = 0
        if padding_type == 'reflect':
            self.pad = nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            self.pad = nn.ReplicationPad2d(1)
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=bias)
        self.norm1 = norm_layer(dim)
        self.act = nn.ReLU()
        self.dropout = nn.Identity()
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=bias)
        self.norm2 = norm_layer(dim)
        
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.pad(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.pad(x)
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = x + out
        return out        

    

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
        self.act = nn.ReLU()
        self.norm = norm_layer(out_dim)
        
    def forward(self, x):
        """Forward function (with skip connections)"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x        


class UpConvBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_dim, out_dim, norm_layer, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(UpConvBlock, self).__init__()

        
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)
        self.act = nn.ReLU()
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """Forward function (with skip connections)"""
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x        
