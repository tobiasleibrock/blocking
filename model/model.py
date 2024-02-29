### ML ###
import datetime
import logging
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from torch import nn
import numpy as np
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from sklearn.model_selection import KFold
from torchvision.models.inception import Inception3
from util import (
    get_dataset,
    get_optimizer,
    get_scheduler,
    get_model,
    get_transform,
    get_date,
)

### CUSTOM MODULES ###
from dataset import GeoEra5Dataset, GeoUkesmDataset, TransformDataset

#### UKESM #####
# (1): [{'model': 'efficientnet_m', 'scheduler': 'step_01', 'loss': 'bce', 'sampler': 'none', 'augmentation': 'light',
# 'lr': '6.67E-3', 'batch_size': 121, 'optimizer': 'sgd_0', 'dropout': '2.18E-1', 'weight_decay': '2.26E-1'}, loss 2.88E-1, island 0, worker 6, generation 2]
# INFO = "runs_ukesm/20_02_2024/inc/step_01/bce_weighted/no_sa/light/lr0.00667/b121/adagrad/d0.118/wd0.126/"
# BATCH_SIZE = 121
# LEARNING_RATE = 0.00667
# DROPOUT = 0.118
# MOMENTUM = 0.0
# WEIGHT_DECAY = 0.126
# EPOCHS = 30
# FOLDS = 101
# TRANSFORM = "light"  # "heavy", "light", "none"
# LOSS = "bce_weighted"  # "bce_weighted", "bce"
# OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
# SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
# MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
# DEBUG = True
# TRAIN_DATASET = "ukesm"
# TEST_DATASET = "ukesm"

#### ERA5 #####
# (1): [{'model': 'inception', 'scheduler': 'step_01', 'loss': 'bce_weighted', 'sampler': 'none', 'augmentation': 'heavy',
# 'lr': '4.19E-3', 'batch_size': 157, 'optimizer': 'adagrad', 'dropout': '5.06E-3', 'weight_decay': '4.70E-1'}, loss 2.33E-1, island 0, worker 5, generation 0]
# INFO = "runs_era5/20_02_2024/inc/step_01/bce_weighted/no_sa/heavy/lr0.00419/b157/adagrad/d0.005/wd0.470/"
# BATCH_SIZE = 157
# LEARNING_RATE = 0.00419
# DROPOUT = 0.00506
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.47
# EPOCHS = 30
# FOLDS = 41
# TRANSFORM = "heavy"  # "heavy", "light"
# LOSS = "bce_weighted"  # "bce_weighted", "bce"
# OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
# SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
# MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
# DEBUG = True
# TRAIN_DATASET = "era5"
# TEST_DATASET = "era5"

#### ERA5 MSL #####
# [{'model': 'efficientnet_m', 'scheduler': 'plateau', 'loss': 'bce_weighted', 'sampler': 'none', 'augmentation': 'heavy', 'lr': '7.33E-3',
# 'batch_size': 83, 'optimizer': 'sgd_0', 'dropout': '2.12E-1', 'weight_decay': '2.06E-1'}, loss 3.51E-1, island 0, worker 4, generation 1]
INFO = "runs_era5_msl/21_02_2024/eff_m/plateau/bce_weighted/no_sa/heavy/lr0.00733/b83/sgd_0/d0.212/wd0.206/"
BATCH_SIZE = 83
LEARNING_RATE = 0.00733
DROPOUT = 0.212
MOMENTUM = 0.0
WEIGHT_DECAY = 0.206
EPOCHS = 30
FOLDS = 41
TRANSFORM = "heavy"  # "heavy", "light"
LOSS = "bce_weighted"  # "bce_weighted", "bce"
OPTIMIZER = "sgd_0"  # "adagrad", "adam", "sgd_09", "sgd_0"
SCHEDULER = "plateau"  # "step_01", "step_09", "plateau", "none"
MODEL = "efficientnet_m"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
DEBUG = True
TRAIN_DATASET = "era5-msl"
TEST_DATASET = "era5-msl"

#### EXPERIMENTATION #####
# (1): [{'model': 'inception', 'scheduler': 'step_01', 'loss': 'bce_weighted', 'sampler': 'none', 'augmentation': 'heavy',
# 'lr': '4.19E-3', 'batch_size': 157, 'optimizer': 'adagrad', 'dropout': '5.06E-3', 'weight_decay': '4.70E-1'}, loss 2.33E-1, island 0, worker 5, generation 0]
# INFO = "experimentation"
# BATCH_SIZE = 128
# LEARNING_RATE = 0.005
# DROPOUT = 0.1
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.1
# EPOCHS = 30
# FOLDS = 101
# TRANSFORM = "light"  # "heavy", "light", "none"
# LOSS = "bce_weighted"  # "bce_weighted", "bce"
# OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
# SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
# MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
# DEBUG = False
# TRAIN_DATASET = "ukesm"
# TEST_DATASET = "ukesm"

print("CONFIGURATION")
print(
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT,
    MOMENTUM,
    WEIGHT_DECAY,
    EPOCHS,
    FOLDS,
    INFO,
    TRANSFORM,
    LOSS,
    OPTIMIZER,
    SCHEDULER,
    MODEL,
    DEBUG,
    TRAIN_DATASET,
    TEST_DATASET,
)

full_test_labels = torch.tensor([])
full_test_outputs = torch.tensor([])


def train_model(
    model, optimizer, scheduler, train_dataset, test_dataset, fold, num_epochs
):
    print(f"fold {fold}/{FOLDS}")

    if DEBUG:
        year = get_date(test_dataset[0][2], TEST_DATASET).year
        test_writer = SummaryWriter(f"{INFO}/te/{str(year)}/{fold}")
        training_writer = SummaryWriter(f"{INFO}/tr/{str(year)}/{fold}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_labels = torch.tensor([])
    test_outputs = torch.tensor([])

    for epoch in range(num_epochs):
        print(f"epoch {epoch+1}/{num_epochs}")
        ### TRAINING ###
        model.train()
        epoch_loss = 0.0
        epoch_labels = torch.tensor([])
        epoch_outputs = torch.tensor([])
        for inputs, labels, time in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # fix for inception model in pytorch
            # https://discuss.pytorch.org/t/inception-v3-is-not-working-very-well/38296/3
            if type(model) is Inception3:
                outputs = model(inputs.float())[0]
            else:
                outputs = model(inputs.float())

            if LOSS == "bce_weighted":
                class_counts = torch.bincount(labels.long())
                class_weights = BATCH_SIZE / (2.0 * class_counts.float())
                sample_weights = class_weights[labels.long()]
                criterion = nn.BCELoss(weight=sample_weights)
            else:
                criterion = nn.BCELoss()

            loss = criterion(outputs.flatten(), labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += outputs.shape[0] * loss.item()
            epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
            epoch_outputs = torch.cat(
                (epoch_outputs, outputs.flatten().detach().cpu()), 0
            )

        epoch_loss = epoch_loss / len(epoch_labels)
        print(f"trn f1 {f1(epoch_outputs, epoch_labels)}")

        if DEBUG:
            training_writer.add_scalar("loss", epoch_loss, epoch)
            training_writer.add_scalar(
                "recall", recall(epoch_outputs, epoch_labels), epoch
            )
            training_writer.add_scalar(
                "precision", precision(epoch_outputs, epoch_labels), epoch
            )
            training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)

        ### TEST ###
        model.eval()
        epoch_loss = 0.0
        epoch_labels = torch.tensor([])
        epoch_outputs = torch.tensor([])
        with torch.no_grad():
            for inputs, labels, time in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                epoch_loss += outputs.shape[0] * loss.item()
                epoch_labels = torch.cat(
                    (epoch_labels, labels.float().detach().cpu()), 0
                )
                epoch_outputs = torch.cat(
                    (epoch_outputs, outputs.flatten().detach().cpu()), 0
                )

        epoch_loss = epoch_loss / len(epoch_labels)
        print(f"tst f1 {f1(epoch_outputs, epoch_labels)}")
        date_from = get_date(test_dataset[0][2], TEST_DATASET)
        date_to = get_date(test_dataset[-1][2], TEST_DATASET)

        print(f"test from {date_from} to {date_to}")
        print("-----------------------------------")

        if DEBUG:
            test_writer.add_scalar("loss", epoch_loss, epoch)
            test_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
            test_writer.add_scalar(
                "precision", precision(epoch_outputs, epoch_labels), epoch
            )
            test_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)

        # CONFUSION MATRIX
        if DEBUG:
            conf_matrix = confusion_matrix(
                epoch_labels,
                (epoch_outputs >= 0.5).int(),
                labels=np.array([0, 1]),
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix,
                display_labels=["no blocking", "blocking"],
            )
            disp.plot()
            test_writer.add_figure("conf-matrix", disp.figure_, global_step=epoch)

        if epoch == num_epochs - 1:
            print(
                f" epoch shape {epoch_labels.shape} {epoch_outputs.shape} test shape {test_labels.shape} {test_outputs.shape}"
            )
            test_labels = torch.cat((test_labels, epoch_labels), 0)
            test_outputs = torch.cat((test_outputs, epoch_outputs), 0)

        if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
            scheduler.step(loss.item())
        else:
            scheduler.step()

    return model, test_labels, test_outputs


device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

tr_dataset = get_dataset(TRAIN_DATASET)
ts_dataset = get_dataset(TEST_DATASET)

kf = KFold(n_splits=FOLDS, shuffle=False)

for fold, (train_indices, test_indices) in enumerate(kf.split(tr_dataset)):
    model = get_model(MODEL, DROPOUT)
    model.to(device)

    transform = get_transform(TRANSFORM)
    optimizer = get_optimizer(OPTIMIZER, WEIGHT_DECAY, LEARNING_RATE, model)
    scheduler = get_scheduler(SCHEDULER, optimizer)

    train_ds = Subset(tr_dataset, train_indices)

    subset_data = [tr_dataset[idx] for idx in train_ds.indices]
    _, subset_labels, _ = zip(*subset_data)
    labels = torch.tensor(subset_labels).long()
    train_counts = torch.bincount(labels)
    train_class_weights = len(labels) / (2.0 * train_counts.float())
    train_weights = train_class_weights[labels]
    train_sampler = WeightedRandomSampler(train_weights, len(labels))

    # train_ds = ConcatDataset([train_ds, ukesm_dataset])
    train_ds = TransformDataset(subset_data, transform=transform)

    result = train_model(
        model,
        optimizer,
        scheduler,
        train_ds,
        Subset(tr_dataset, test_indices),
        fold,
        EPOCHS,
    )

    if result:
        full_test_labels = torch.cat((full_test_labels, result[1]), 0)
        full_test_outputs = torch.cat((full_test_outputs, result[2]), 0)
        print("RUNNING METRICS")
        print(full_test_labels.shape, full_test_outputs.shape)
        print(f"f1: {f1(full_test_outputs, full_test_labels)}")
        print(f"recall: {recall(full_test_outputs, full_test_labels)}")
        print(f"precision: {precision(full_test_outputs, full_test_labels)}")

print("FINAL METRICS")
print(INFO)
print(f"f1: {f1(full_test_outputs, full_test_labels)}")
print(f"recall: {recall(full_test_outputs, full_test_labels)}")
print(f"precision: {precision(full_test_outputs, full_test_labels)}")
