import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Iterable, Callable

class Extractor(nn.Module):
    
    """
    usage:
    feature_list = ['features_13','features_23','features_33','features_43']
    extract_model = Extractor(model=pretrained_model,layers=feature_list).cuda()
    """
    
    def __init__(self, model: nn.Module=None, layers: Iterable[str]=None):
        super().__init__()
        self.model = model
        self._features = {}
        if self.model == None :
            self.model =  models.vgg16(pretrained=True)
        self.model = self.model.eval()
  
        for param in self.model.parameters():
            param.requires_grad = False

        for name, layer in self.model.named_children():
            
            if isinstance(layer, nn.Sequential):
                for children_name, children_layer in layer.named_children():
                    children_layer.__name__ = name + '_' + children_name

                    if children_layer.__name__ in layers:
                        children_layer.register_forward_hook(
                            self.save_output()
                        )
                        
            else:
                layer.__name__ = name
                if layer.__name__ in layers: 
                    layer.register_forward_hook(
                        self.save_output()
                    )
                    
    def save_output(self):
        def fn(layer, _, output):
            self._features[layer.__name__] = output
        return fn
    
    def forward(self, x):
        self.model = self.model.to(x.device)
        _ = self.model(x)
        return self._features