import torch
import torch.nn as nn
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s


def get_model(dropout):
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT, dropout=dropout)

    # create new first conv layer (resnet)
    weights = model.features[0][0].weight.clone()
    model.features[0][0] = nn.Conv2d(
        5, 24, kernel_size=3, stride=2, padding=1, bias=False
    )
    with torch.no_grad():
        model.features[0][0].weight[:, 0] = weights[:, 0]
        model.features[0][0].weight[:, 1] = weights[:, 0]
        model.features[0][0].weight[:, 2] = weights[:, 0]
        model.features[0][0].weight[:, 3] = weights[:, 0]
        model.features[0][0].weight[:, 4] = weights[:, 0]

    # create new final linear layer
    fully_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(nn.Linear(fully_features, 1), nn.Sigmoid())

    for index, parameter in enumerate(model.parameters()):
        if index < 220:
            parameter.requires_grad = False

    model.float()

    return model