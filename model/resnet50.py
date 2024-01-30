import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def get_model(linear_only):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # create new first conv layer (resnet)
    conv1_weights = model.conv1.weight.clone()
    model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight[:, :3] = conv1_weights
        model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
        model.conv1.weight[:, 4] = model.conv1.weight[:, 0]

    # disable gradient descent for all pre-trained parameters
    for parameters in model.parameters():
        parameters.requires_grad = not linear_only

    # create new final linear layer
    fully_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fully_features, 1), nn.Sigmoid())

    model.float()

    return model
