import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR

def Scheduler(optimizer, name, **kwargs):

        if name == 'cosineAnn':
                return CosineAnnealingLR(optimizer, **kwargs)
        elif name == 'cosineAnnWarm':
                return CosineAnnealingWarmRestarts(optimizer, **kwargs)
        elif callable(name):
                return name(params, **kwargs)

        else:
                raise ValueError('Scheduler should be callable/cosineAnn/cosineAnnWarm; got {}'.format(name))


