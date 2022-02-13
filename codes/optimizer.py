import torch
import torch.nn as nn
from adabelief_pytorch import AdaBelief

def Optimizer(params, name, **kwargs):

        name = name.lower()
        
        if name == "none" or name == "identity" or name is None:
                return torch.nn.Identity(**kwargs)
        elif name == 'adam':
                return torch.optim.Adam(params, **kwargs)
        elif name == 'sgd':
                return torch.optim.SGD(params, **kwargs)
        elif name == "adabelief":
                return AdaBelief(params, lr=2e-4, eps=1e-12, betas=(0.9,0.999), weight_decouple = True, rectify = False)
        elif callable(name):
                return name(params, **kwargs)

        else:
                raise ValueError('Optimizer should be callable/adam/sgd; got {}'.format(name))

                
