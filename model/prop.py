import random
from matplotlib.artist import get
import torch
import logging


from mpi4py import MPI

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from resnet18 import get_model as get_resnet18_model
from resnet50 import get_model as get_resnet50_model
from inception_v3 import get_model as get_inception_model
from efficientnet_s import get_model as get_efficientnet_s_model
from dataset import BlockingObservationalDataset1x1

from propulate.utils import get_default_propagator, set_logger_config
from propulate.propulator import Propulator

import os
from datetime import datetime
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from torchvision.models.inception import Inception3

NUM_EPOCHS = 40
NUM_FOLDS = 5
TENSORBOARD_PREFIX = "runs_propulate/"
DEBUG = False

models = {
    "resnet18": get_resnet18_model,
    "resnet50": get_resnet50_model,
    "efficientnet_s": get_efficientnet_s_model,
    "inception": get_inception_model,
}

optimizers = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "adagrad": optim.Adagrad,
}

def get_scheduler(key, optimizer):
    schedulers = {
        "step_02": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2),
        "step_01": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
        "step_05": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5),
        "none": None,
    }
    return schedulers[key]

def ind_loss(params):
    rank = MPI.COMM_WORLD.rank
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    f1 = BinaryF1Score(threshold=0.5).to(device)
    recall = BinaryRecall(threshold=0.5).to(device)
    precision = BinaryPrecision(threshold=0.5).to(device)

    current_datetime = datetime.now()
    run = current_datetime.strftime("%d-%H-%M")

    mean_f1 = np.zeros(NUM_EPOCHS)
    mean_loss = np.zeros(NUM_EPOCHS)

    if DEBUG:
        validation_writer = SummaryWriter(f"{TENSORBOARD_PREFIX}{run}/validation/{params['model']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']}/{params['weight_decay']}")
        training_writer = SummaryWriter(f"{TENSORBOARD_PREFIX}{run}/training/{params['model']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']}/{params['weight_decay']}")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True)

    count_folds = 0
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        # if fold > 0: continue
        count_folds += 1

        model = models[params["model"]](dropout=params["dropout"])
        model.to(device)

        if params["optimizer"] == "sgd_0":
            optimizer = optimizers["sgd"](
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], momentum=0
            )
        elif params["optimizer"] == "sgd_09":
            optimizer = optimizers["sgd"](
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], momentum=0.9
            )
        else:
            optimizer = optimizers[params["optimizer"]](
                model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
            )

        scheduler = get_scheduler(params["scheduler"], optimizer)

        train_ds = Subset(train_dataset, train_indices)

        subset_data = [train_dataset[idx] for idx in train_ds.indices]
        _, subset_labels, _ = zip(*subset_data)
        labels = torch.tensor(subset_labels).long()
        train_counts = torch.bincount(labels)
        train_class_weights = len(labels) / (2.0 * train_counts.float())
        train_weights = train_class_weights[labels]
        train_sampler = WeightedRandomSampler(train_weights, len(labels))

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=params["batch_size"], shuffle=False, sampler=train_sampler
        )
        val_loader = torch.utils.data.DataLoader(
            Subset(train_dataset, val_indices), batch_size=params["batch_size"], shuffle=False
        )

        for epoch in range(NUM_EPOCHS):
            ### TRAINING ###

            model.train()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # fix for inception model in pytorch
                # https://discuss.pytorch.org/t/inception-v3-is-not-working-very-well/38296/3
                if type(model) is Inception3:
                    outputs = model(inputs.float())[0]
                else:
                    outputs = model(inputs.float())

                if params["loss"] == "bce":
                    criterion = nn.BCELoss()
                elif params["loss"] == "bce_weighted":
                    # scale loss weights by class imbalance in input data
                    class_counts = torch.bincount(labels.long())
                    class_weights = params["batch_size"] / (2.0 * class_counts.float())
                    sample_weights = class_weights[labels.long()]
                    criterion = nn.BCELoss(weight=sample_weights)

                loss = criterion(outputs.flatten(), labels.float())
                loss.backward()
                optimizer.step()

                epoch_loss += outputs.shape[0] * loss.item()
                epoch_labels = torch.cat(
                    (epoch_labels, labels.float().detach().cpu()), 0
                )
                epoch_outputs = torch.cat(
                    (epoch_outputs, outputs.flatten().detach().cpu()), 0
                )

            epoch_loss = epoch_loss / len(epoch_labels)
            epoch_predictions = (epoch_outputs > 0.5).float()

            # print(f"epoch {epoch + 1}/{NUM_EPOCHS}")
            # print("training")
            # print((torch.bincount(epoch_predictions.long()), torch.bincount(epoch_labels.long())))
            # print(f"training outputs: {epoch_outputs[:9]}")
            # print(f"t_gue: {epoch_predictions[:17]}") 
            # print(f"t_lab: {epoch_labels[:17]}")
            # print(f"training {fold} f1: {f1(epoch_predictions, epoch_labels)} loss: {epoch_loss} mean: {torch.mean(epoch_predictions)}")

            if DEBUG:
                training_writer.add_scalar("loss", epoch_loss, epoch)
                training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
                training_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
                training_writer.add_scalar("precision", precision(epoch_outputs, epoch_labels), epoch)

            ### VALIDATION ###
            model.eval()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            with torch.no_grad():
                for inputs, labels, t in val_loader:
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
            epoch_predictions = (epoch_outputs > 0.5).float()

            mean_f1[epoch] += f1(epoch_predictions, epoch_labels)


            if len(torch.bincount(epoch_predictions.long())) == 0:
                print((epoch_outputs, epoch_predictions, epoch_labels))
            
            # print("validation")
            # print((torch.bincount(epoch_predictions.long()), torch.bincount(epoch_labels.long())))
            # print(f"validation outputs: {epoch_outputs[:9]}")
            # print(f"v_gue: {epoch_predictions[:17]}") 
            # print(f"v_lab: {epoch_labels[:17]}")
            # print(f"validation fold {fold} f1: {f1(epoch_predictions, epoch_labels)} loss: {epoch_loss} mean: {torch.mean(epoch_predictions)}")
            # print()
            # print()
            # print()
            mean_loss[epoch] += epoch_loss

            if DEBUG:
                validation_writer.add_scalar("loss", epoch_loss, epoch)
                validation_writer.add_scalar("f1", f1(epoch_predictions, epoch_labels), epoch)
                validation_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
                validation_writer.add_scalar("precision", precision(epoch_outputs, epoch_labels), epoch)

            if scheduler:
                if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(loss.item())
                else:
                    scheduler.step()

    mean_loss = np.divide(mean_loss, count_folds)
    mean_f1 = np.divide(mean_f1, count_folds)

    if DEBUG:
        for idx, (loss, f1) in enumerate(zip(mean_loss, mean_f1)):
            validation_writer.add_scalar("loss", mean_loss[idx], idx)
            validation_writer.add_scalar("f1", mean_f1[idx], idx)

    return 1 - mean_f1[-1]


set_logger_config(
    level=logging.INFO,  # logging level
    log_file=f"./propulate.log",  # logging path
    log_to_stdout=True,  # Print log on stdout.
    log_rank=False,  # Do not prepend MPI rank to logging messages.
    colors=False,  # Use colors.
)

era5_dataset = BlockingObservationalDataset1x1()

test_size = int(len(era5_dataset) * 0.15)
train_size = len(era5_dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(
    era5_dataset, [train_size, test_size]
)

comm = MPI.COMM_WORLD
num_generations = 10
pop_size = 2 * MPI.COMM_WORLD.size
# limits = {
#     "model": ("resnet18", "resnet50", "efficientnet_s", "inception"),
#     "scheduler": ("step", "plateau", "none"),
#     "loss": ("bce", "bce_weighted"),
#     "lr": (0.05, 0.0001),
#     "batch_size": (4, 256),
#     "optimizer": ("sgd_0", "sgd_09", "adam", "adagrad"),
#     "dropout": (0.0, 0.8),
#     "weight_decay": (0.0, 5.0),
# }
limits = {
    "model": ("resnet18", "resnet50", "inception", "efficientnet_s"),
    "scheduler": ("step_01", "step_02"),
    "loss": ("bce", "bce"),
    "lr": (0.05, 0.001),
    "batch_size": (64, 256),
    "optimizer": ("adagrad", "sgd_09"),
    "dropout": (0.0, 0.5),
    "weight_decay": (0.0, 1.0),
}
rng = random.Random(MPI.COMM_WORLD.rank)
# hyperparameters from https://propulate.readthedocs.io/en/latest/tut_hpo.html
propagator = get_default_propagator(
    pop_size=pop_size,
    limits=limits,
    mate_prob=0.7,
    mut_prob=0.4,
    random_prob=0.1,
    rng=rng,
)

propulator = Propulator(
    propagator=propagator,
    loss_fn=ind_loss,
    comm=comm,
    generations=num_generations,
    rng=rng,
)

propulator.propulate(1, 2)
propulator.summarize(top_n=pop_size, debug=2)
