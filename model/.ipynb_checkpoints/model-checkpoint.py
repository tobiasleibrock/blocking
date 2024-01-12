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

BATCH_SIZE = 64
LEARNING_RATE = 0.001

model = get_resnet18_model(linear_only=False)
#model = get_resnet50_model(linear_only=True)
#model = get_efficientnet_model(linear_only=False)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

dataset = BlockingObservationalDataset()
test_size = len(dataset) * 0.15
train_size = len(dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

training_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

dataloaders = {"train": training_loader, "val": test_loader}

train_model(model, criterion, optimizer, scheduler, dataloaders, 30)
