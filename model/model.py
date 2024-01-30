import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from alexnet import get_model as get_alexnet_model
from dataset import BlockingObservationalDataset1x1, BlockingUKESMDataset1x1
from train import train_model

LEARNING_RATE = 0.001

print("setting up model..")

# model = get_resnet18_model()
model = get_alexnet_model()
# model = get_resnet50_model(linear_only=True)
# model = get_efficientnet_model(linear_only=False)

print("setting up data..")

era5_dataset = BlockingObservationalDataset1x1()
ukesm_dataset = BlockingUKESMDataset1x1()
# dataset = BlockingObservationalDataset()

test_size = int(len(era5_dataset) * 0.15)
train_size = len(era5_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(
    era5_dataset, [train_size, test_size]
)

datasets = {"train": train_dataset, "test": test_dataset, "ukesm": ukesm_dataset}
# samplers = {"train": train_sampler, "ukesm": ukesm_sampler}

optimizer = optim.SGD(
    model.parameters(), lr=LEARNING_RATE, momentum=0.1, weight_decay=0.01
)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
train_model(model, optimizer, scheduler, datasets, f"wd-{0.01}-alex", 30)
