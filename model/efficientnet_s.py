import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def get_model(linear_only):
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # create new first conv layer (resnet)
    weights = model.features[0][0].weight.clone()
    model.features[0][0] = nn.Conv2d(
        5, 24, kernel_size=3, stride=2, padding=1, bias=False
    )
    with torch.no_grad():
        model.features[0][0].weight[:, :3] = weights
        model.features[0][0].weight[:, 3] = model.features[0][0].weight[:, 0]
        model.features[0][0].weight[:, 4] = model.features[0][0].weight[:, 0]

    # disable gradient descent for all pre-trained parameters
    for parameters in model.parameters():
        parameters.requires_grad = not linear_only

    # create new final linear layer
    fully_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Linear(fully_features, 1), nn.Sigmoid())

    model.float()

    return model
