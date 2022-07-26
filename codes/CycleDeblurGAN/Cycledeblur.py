import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=1, stride=1, padding=0, norm='in', activation='silu'):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding)

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=1, stride=2, padding=0, output_padding=0, norm='in', activation='silu'):
        super(UpConvBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=padding, output_padding=output_padding)

        if norm == "bn":
            self.norm = nn.BatchNorm2d(out_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(out_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
    
class InceptionBlock(nn.Module):
    def __init__(self, dim, stride=1, norm='in', activation='silu'):
        super(InceptionBlock, self).__init__()

        self.inception11 = ConvBlock(dim // 2, dim, 1, stride, 0, norm=norm, activation=activation)
        self.inception55 = ConvBlock(dim // 2, dim, 5, stride, 5 // 2, norm=norm, activation=activation)
        self.inception99 = ConvBlock(dim // 2, dim, 9, stride, 9 // 2, norm=norm, activation=activation)
        self.inception1313 = ConvBlock(dim // 2, dim, 13, stride, 13 // 2, norm=norm, activation=activation)
        self.conv = nn.Conv2d(dim*4, dim, 1)

    def forward(self, x):
        x11 = self.inception11(x)
        x55 = self.inception55(x)
        x99 = self.inception99(x)
        x1313 = self.inception1313(x)
        x = torch.cat([x11, x55, x99, x1313], dim=1)
        x = self.conv(x)
        return x

    
    
class ResBlock(nn.Module):
    def __init__(self, dim, kernel=3, stride=1, padding=1, norm='in', activation='silu'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(dim, dim, kernel, stride=stride, padding=padding)

        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        elif norm == "in":
            self.norm1 = nn.InstanceNorm2d(dim)
            self.norm2 = nn.InstanceNorm2d(dim)
        elif norm == "none":
            self.norm1 = None
            self.norm2 = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = residual + x
        return x
    
    

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, activation="sigmoid"):
        super(Generator, self).__init__()
        dim = 64
        self.e_layer1 = ConvBlock(in_channels, dim, kernel=7, stride=1, padding=7//2, norm='in', activation='silu')
        dim *= 2
        self.e_layer2 = InceptionBlock(dim, stride=2, norm='in', activation='silu')
        dim *= 2
        self.e_layer3 = InceptionBlock(dim, stride=2, norm='in', activation='silu')
        
        resblocks = []
        for _ in range(9):
            resblocks += [ResBlock(dim, kernel=3, stride=1, padding=1, norm='in', activation='silu')]
        self.e_layer4 = nn.Sequential(*resblocks)
        
        self.d_layer3 = UpConvBlock(dim, dim//2, kernel=3, stride=2, padding=3//2, output_padding=3//2, norm='in', activation='silu')
        dim //= 2
        self.d_layer2 = UpConvBlock(dim, dim//2, kernel=3, stride=2, padding=3//2, output_padding=3//2, norm='in', activation='silu')
        dim //= 2
        self.d_layer1 = ConvBlock(dim, out_channels, kernel=7, stride=1, padding=7//2, norm='none', activation=activation)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=False)
        elif activation == "silu":
            self.activation = nn.SiLU(inplace=False)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
            

    def forward(self, x):
        e1 = self.e_layer1(x)
        e2 = self.e_layer2(e1)
        e3 = self.e_layer3(e2)
        e4 = self.e_layer4(e3)
        x = e4 + e3
        e3 = self.d_layer3(x)
        e3 = e3 + e2
        e2 = self.d_layer2(e3) + e1
        e1 = self.d_layer1(e2)
        
        return e1


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, activation="sigmoid"):
        super(Discriminator, self).__init__()
        
        dim = 64
        e_layer1 = [ConvBlock(in_channels, dim, kernel=4, stride=1, padding=4//2, norm='none', activation='silu'),
                         ConvBlock(dim, dim, kernel=4, stride=2, padding=4//2, norm='in', activation='silu')]
        self.e_layer1 = nn.Sequential(*e_layer1)
        dim *= 2
        e_layer2 = [ConvBlock(dim//2, dim, kernel=4, stride=1, padding=4//2, norm='in', activation='silu'),
                    ConvBlock(dim, dim, kernel=4, stride=2, padding=4//2, norm="in", activation="silu")]
        self.e_layer2 = nn.Sequential(*e_layer2)
        dim *= 2
        e_layer3 = [ConvBlock(dim//2, dim, kernel=4, stride=1, padding=4//2, norm="in", activation="silu"),
                   ConvBlock(dim, out_channels, kernel=4, stride=1, padding=4//2, norm="none", activation="silu")]
        self.e_layer3 = nn.Sequential(*e_layer3)
        self.pooling = nn.AvgPool2d(2)
        

    def forward(self, x):
        x = self.e_layer1(x)
        x = self.e_layer2(x)
        x = self.e_layer3(x)
        x = self.pooling(x)
        return x

