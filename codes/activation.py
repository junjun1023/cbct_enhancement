import torch
import torch.nn as nn


def Activation(name, **params):
    if name is None or name == 'identity':
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax2d':
        return nn.Softmax(dim=1, **params)
    elif name == 'softmax':
        return nn.Softmax(dim=1, **params)
    elif name == 'logsoftmax':
         return nn.LogSoftmax(**params)
    elif name == 'tanh':
        return nn.Tanh()
    elif name== "hardtanh":
        return nn.Hardtanh(min_val=0, max_val=1)
    elif name == 'argmax':
        return ArgMax(**params)
    elif name == 'argmax2d':
        return ArgMax(dim=1, **params)
    elif callable(name):
        return name(**params)
    else:
        raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))


