import torch
import torch.nn as nn
import torchvision.models as models

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.model = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            features[name] = x
        return features