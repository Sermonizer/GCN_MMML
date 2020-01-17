import json
import torch
import torch.nn as nn
import torchvision.models as models

model = models.resnet101(pretrained=True)
feature = torch.nn.Sequential(*list(model.children())[:])
print(feature)

# def EncodeImage():
