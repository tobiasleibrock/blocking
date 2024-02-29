### ML ###
import datetime
import logging
from pickle import FALSE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch
from torch import nn
import numpy as np
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from sklearn.model_selection import KFold
from torchvision.models.inception import Inception3
from util import get_optimizer, get_scheduler, get_model, get_transform

### CUSTOM MODULES ###
from dataset import GeoEra5Dataset5D, GeoUkesmDataset5D, TransformDataset
from dataset import GeoEra5Dataset, GeoUkesmDataset, TransformDataset


#### Train: UKESM Test: ERA5 #####
# (1): [{'model': 'efficientnet_m', 'scheduler': 'step_01', 'loss': 'bce', 'sampler': 'none', 'augmentation': 'light',
# 'lr': '6.67E-3', 'batch_size': 121, 'optimizer': 'sgd_0', 'dropout': '2.18E-1', 'weight_decay': '2.26E-1'}, loss 2.88E-1, island 0, worker 6, generation 2]
# INFO = "runs_ukesm_t_era5/21_02_2024/inc/step_01/bce_weighted/no_sa/light/lr0.00667/b121/adagrad/d0.118/wd0.126/"
# SINFO = "ukesm-geo-final"
# BATCH_SIZE = 121
# LEARNING_RATE = 0.00667
# DROPOUT = 0.118
# MOMENTUM = 0.0
# WEIGHT_DECAY = 0.126
# EPOCHS = 30
# TRANSFORM = "light"  # "heavy", "light", "none"
# LOSS = "bce_weighted"  # "bce_weighted", "bce"
# OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
# SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
# MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
# DEBUG = True
# TRAIN_DATASET = "ukesm"
# TEST_DATASET = "era5"

#### Train: ERA5 Test: UKESM #####
# (1): [{'model': 'inception', 'scheduler': 'step_01', 'loss': 'bce_weighted', 'sampler': 'none', 'augmentation': 'heavy',
# 'lr': '4.19E-3', 'batch_size': 157, 'optimizer': 'adagrad', 'dropout': '5.06E-3', 'weight_decay': '4.70E-1'}, loss 2.33E-1, island 0, worker 5, generation 0]
INFO = "runs_era5_t_ukesm/21_02_2024/inc/step_01/bce_weighted/no_sa/heavy/lr0.00419/b157/adagrad/d0.005/wd0.470/"
SINFO = "era5-geo-final"
BATCH_SIZE = 157
LEARNING_RATE = 0.00419
DROPOUT = 0.00506
MOMENTUM = 0.9
WEIGHT_DECAY = 0.47
EPOCHS = 30
TRANSFORM = "heavy"  # "heavy", "light"
LOSS = "bce_weighted"  # "bce_weighted", "bce"
OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"
DEBUG = False
TRAIN_DATASET = "era5"
TEST_DATASET = "ukesm"

print("CONFIGURATION")
print(
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT,
    MOMENTUM,
    WEIGHT_DECAY,
    EPOCHS,
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

# handler = logging.FileHandler(f"{INFO}.log")
# handler.setFormatter(logging.Formatter("%(message)s"))
# csv_logger = logging.getLogger("csv_logger")
# csv_logger.addHandler(handler)
# csv_logger.setLevel(logging.INFO)

# csv_logger.info("date,label,prediction,output")


def get_date(offset, dataset):
    if dataset == "era5":
        return datetime.datetime(1900, 1, 1) + datetime.timedelta(hours=int(offset))
    if dataset == "ukesm":
        reference_date = datetime.datetime(1850, 1, 1)
        return reference_date + datetime.timedelta(days=int(offset / 360 * 365.25))


def train_model(model, optimizer, scheduler, train_dataset, test_dataset, num_epochs):
    if DEBUG:
        year = get_date(test_dataset[0][2], TEST_DATASET).year
        test_writer = SummaryWriter(f"{INFO}/te/{str(year)}/")
        training_writer = SummaryWriter(f"{INFO}/tr/{str(year)}/")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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

        scheduler.step()

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
            epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
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

    print("TEST METRICS")
    print(INFO)
    print(f"f1: {f1(epoch_outputs, epoch_labels)}")
    print(f"recall: {recall(epoch_outputs, epoch_labels)}")
    print(f"precision: {precision(epoch_outputs, epoch_labels)}")

    return model


device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

tr_dataset = GeoEra5Dataset5D() if TRAIN_DATASET == "era5" else GeoUkesmDataset5D()
ts_dataset = GeoEra5Dataset5D() if TEST_DATASET == "era5" else GeoUkesmDataset5D()
# tr_dataset = GeoEra5Dataset() if TRAIN_DATASET == "era5" else GeoUkesmDataset()
# ts_dataset = GeoEra5Dataset() if TEST_DATASET == "era5" else GeoUkesmDataset()

model = get_model(MODEL, DROPOUT)
model.to(device)

transform = get_transform(TRANSFORM)
optimizer = get_optimizer(OPTIMIZER, WEIGHT_DECAY, LEARNING_RATE, model)
scheduler = get_scheduler(SCHEDULER, optimizer)

model = train_model(
    model,
    optimizer,
    scheduler,
    tr_dataset,
    ts_dataset,
    EPOCHS,
)

torch.save(model.state_dict(), f"./models/{SINFO}.pt")
