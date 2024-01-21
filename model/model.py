from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s
import torchvision
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dataset import BlockingObservationalDataset1x1, BlockingUKESMDataset1x1
from train import train_model
from resnet18 import get_model as get_resnet18_model
from resnet50 import get_model as get_resnet50_model
from efficientnet_s import get_model as get_efficientnet_model
from collections import Counter

LEARNING_RATE = 0.001

print("setting up model..")

model = get_resnet18_model(linear_only=False)
#model = get_resnet50_model(linear_only=True)
#model = get_efficientnet_model(linear_only=False)

print("setting up data..")

dataset = BlockingObservationalDataset1x1()
ukesm_dataset = BlockingUKESMDataset1x1()
#dataset = BlockingObservationalDataset()

test_size = int(len(dataset) * 0.15)
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

print("training_dist: " + str(dict(Counter(train_dataset.dataset.labels))))
print("ukesm_dist: " + str(dict(Counter(ukesm_dataset.labels))))

datasets = {"train": train_dataset, "test": test_dataset, "ukesm": ukesm_dataset}

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.1, weight_decay=0.01)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

train_model(model, optimizer, scheduler, datasets, 40)


















