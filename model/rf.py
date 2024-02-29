### ML ###
import datetime
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
from dataset import GeoUkesmDataset, TransformDataset

#### ERA5 #####
# (1): [{'model': 'inception', 'scheduler': 'step_01', 'loss': 'bce_weighted', 'sampler': 'none', 'augmentation': 'heavy',
# 'lr': '4.19E-3', 'batch_size': 157, 'optimizer': 'adagrad', 'dropout': '5.06E-3', 'weight_decay': '4.70E-1'}, loss 2.33E-1, island 0, worker 5, generation 0]
INFO = "runs_era5/20-02-2024/inc/step_01/bce_weighted/no_sa/heavy/lr0.00419/b157/adagrad/d0.005/wd0.470/"
BATCH_SIZE = 157
LEARNING_RATE = 0.00419
DROPOUT = 0.00506
MOMENTUM = 0.9
WEIGHT_DECAY = 0.47
EPOCHS = 50
FOLDS = 41
TRANSFORM = "heavy"  # "heavy", "light"
LOSS = "bce_weighted"  # "bce_weighted", "bce"
OPTIMIZER = "adagrad"  # "adagrad", "adam", "sgd_09", "sgd_0"
SCHEDULER = "step_01"  # "step_01", "step_09", "plateau", "none"
MODEL = "inception"  # "resnet18", "resnet50", "efficientnet_s", "efficientnet_m", "inception"

DEBUG = True

full_test_labels = torch.tensor([])
full_test_outputs = torch.tensor([])


def train_model(
    model, optimizer, scheduler, train_dataset, test_dataset, fold, num_epochs
):
    print(f"fold {fold}/{FOLDS}")

    if DEBUG:
        year = (
            datetime.datetime(1900, 1, 1)
            + datetime.timedelta(hours=int(test_dataset[0][2]))
        ).year
        test_writer = SummaryWriter(f"{INFO}/te/{str(year)}/{fold}")
        training_writer = SummaryWriter(f"{INFO}/tr/{str(year)}/{fold}")

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
        date_from = datetime.datetime(1900, 1, 1) + datetime.timedelta(
            hours=int(test_loader.dataset[0][2])
        )
        date_to = datetime.datetime(1900, 1, 1) + datetime.timedelta(
            hours=int(test_loader.dataset[-1][2])
        )
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

        scheduler.step()

    return model, epoch_labels, epoch_outputs


device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

# era5_dataset = GeoEra5Dataset()
ukesm_dataset = GeoUkesmDataset()

kf = KFold(n_splits=FOLDS, shuffle=False)

for fold, (train_indices, test_indices) in enumerate(kf.split(ukesm_dataset)):
    model = get_model(MODEL, DROPOUT)
    model.to(device)

    transform = get_transform(TRANSFORM)
    optimizer = get_optimizer(OPTIMIZER, WEIGHT_DECAY, LEARNING_RATE, model)
    scheduler = get_scheduler(SCHEDULER, optimizer)

    train_ds = Subset(ukesm_dataset, train_indices)

    subset_data = [ukesm_dataset[idx] for idx in train_ds.indices]
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
        Subset(ukesm_dataset, test_indices),
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
