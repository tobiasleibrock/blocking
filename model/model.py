from torchvision.models import resnet18, ResNet18_Weights
import torchvision
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset import BlockingObservationalDataset
from train import train_model


model = resnet18(weights=ResNet18_Weights.DEFAULT)

# create new first conv layer
conv1_weights = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    model.conv1.weight[:, :3] = conv1_weights
    model.conv1.weight[:, 3] = model.conv1.weight[:, 0]
    model.conv1.weight[:, 4] = model.conv1.weight[:, 0]

# disable gradient descent for all pre-trained parameters
for parameters in model.parameters():
    parameters.requires_grad = False

# create new final linear layer
fully_features = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(fully_features, 1), nn.ReLU())

model.double()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

training_dataset = BlockingObservationalDataset()
training_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=4, shuffle=True
)

dataloaders = {"train": training_loader, "val": training_loader}

train_model(model, criterion, optimizer, scheduler, dataloaders, 1)
