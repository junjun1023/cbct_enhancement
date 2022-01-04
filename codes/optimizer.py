import torch
import torch.nn as nn


def Optimizer(params, name, **kwargs):

        name = name.lower()
        
        if name == "none" or name == "identity" or name is None:
                return torch.nn.Identity(**kwargs)
        elif name == 'adam':
                return torch.optim.Adam(params, **kwargs)
        elif name == 'sgd':
                return torch.optim.SGD(params, **kwargs)
        elif callable(name):
                return name(params, **kwargs)

        else:
                raise ValueError('Optimizer should be callable/adam/sgd; got {}'.format(name))

