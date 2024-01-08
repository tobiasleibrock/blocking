from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch

def get_model(linear_only):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

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
    print(fully_features)
    model.fc = nn.Sequential(nn.Linear(int(fully_features/4), 1), nn.Sigmoid())
    
    model.float()
    
    return model