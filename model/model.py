from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s
import torchvision
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset import BlockingObservationalDataset
from train import train_model
from resnet18 import get_model as get_resnet18_model
from resnet50 import get_model as get_resnet50_model
from efficientnet_s import get_model as get_efficientnet_model

LEARNING_RATE = 0.001

print("setting up model..")

model = get_resnet18_model(linear_only=False)
#model = get_resnet50_model(linear_only=True)
#model = get_efficientnet_model(linear_only=False)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

print("setting up data..")

dataset = BlockingObservationalDataset()
test_size = int(len(dataset) * 0.15)
train_size = len(dataset) - test_size
print(len(dataset))
print((train_size, test_size))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

datasets = {"train": train_dataset, "test": test_dataset}

train_model(model, criterion, optimizer, scheduler, datasets, 50)
