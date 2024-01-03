from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s
import torchvision
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset import BlockingObservationalDataset
from train import train_model

BATCH_SIZE = 64
LEARNING_RATE = 0.001

#model = resnet18(weights=ResNet18_Weights.DEFAULT)
#model = resnet50(weights=ResNet50_Weights.DEFAULT)
model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

# create new first conv layer (resnet)
#conv1_weights = model.conv1.weight.clone()
#model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
#with torch.no_grad():
#    model.conv1.weight[:, :3] = conv1_weights
#    model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
#    model.conv1.weight[:, 4] = model.conv1.weight[:, 0]
model._conv_stem.in_channels = 5
model._conv_stem.weight = torch.nn.Parameter(torch.cat([model._conv_stem.weight, model._conv_stem.weight], axis=1))

# disable gradient descent for all pre-trained parameters
for parameters in model.parameters():
    parameters.requires_grad = False

# create new final linear layer
fully_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fully_features, 1), nn.Sigmoid())

model.float()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

#scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

training_dataset = BlockingObservationalDataset(run="train")
validation_dataset = BlockingObservationalDataset(run="val")
training_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=BATCH_SIZE, shuffle=False
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=False
)

dataloaders = {"train": training_loader, "val": validation_loader}

train_model(model, criterion, optimizer, None, dataloaders, 15)
