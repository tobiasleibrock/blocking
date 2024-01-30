from torchvision.models import alexnet, AlexNet_Weights
import torch.nn as nn
import torch

def get_model():
    model = alexnet(weights=AlexNet_Weights.DEFAULT)

    # create new first conv layer (resnet)
    conv1_weights = model.features[0].weight.clone()
    model.features[0] = nn.Conv2d(5, 64, kernel_size=11, stride=4, padding=2, bias=False)
    with torch.no_grad():
        model.features[0].weight[:, 1] = conv1_weights[:, 0]
        model.features[0].weight[:, 1] = conv1_weights[:, 0]
        model.features[0].weight[:, 2] = conv1_weights[:, 0]
        model.features[0].weight[:, 3] = conv1_weights[:, 0]
        model.features[0].weight[:, 4] = conv1_weights[:, 0]


    # create new final linear layer
    fully_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Linear(fully_features, 1), 
        nn.Sigmoid()
    )

    for index, parameter in enumerate(model.parameters()):
        if index < 5:
            parameter.requires_grad = False

    model.float()
    
    return model

get_model()