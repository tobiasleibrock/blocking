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

NUM_EPOCHS = 50
NUM_FOLDS = 5
TENSORBOARD_PREFIX = "runs_propulate/"
DEBUG = True


def ind_loss(params):
    rank = MPI.COMM_WORLD.rank
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    # Extract hyperparameter combination to test from input dictionary.
    model = params["model"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    optimizer = params["optimizer"]
    scheduler = params["scheduler"]
    dropout = params["dropout"]
    weight_decay = params["weight_decay"]
    scheduler = params["scheduler"]

    era5_dataset = BlockingObservationalDataset1x1()

    test_size = int(len(era5_dataset) * 0.15)
    train_size = len(era5_dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        era5_dataset, [train_size, test_size]
    )

    models = {
        "resnet18": get_resnet18_model(dropout=dropout),
        "resnet50": get_resnet50_model(dropout=dropout),
        "efficientnet_s": get_efficientnet_s_model(dropout=dropout),
        "inception": get_inception_model(dropout=dropout),
    }

    model = models[model]
    model.to(device)

    optimizers = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
        "adagrad": optim.Adagrad,
    }

    if optimizer == "sgd_0":
        optimizer = optimizers["sgd"](
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0
        )
    elif optimizer == "sgd_09":
        optimizer = optimizers["sgd"](
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        optimizer = optimizers[optimizer](
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    schedulers = {
        "step": lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2),
        "plateau": lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5),
        "none": None,
    }

    scheduler = schedulers[scheduler]

    ### TRAINING MODEL

    f1 = BinaryF1Score(threshold=0.5).to(device)
    # recall = BinaryRecall(threshold=0.5).to(device)
    # precision = BinaryPrecision(threshold=0.5).to(device)

    current_datetime = datetime.now()
    run = current_datetime.strftime("%Y-%m-%d %H:%M")

    mean_f1 = np.zeros(NUM_EPOCHS)
    mean_loss = np.zeros(NUM_EPOCHS)

    if DEBUG:
        validation_writer = SummaryWriter(f"{TENSORBOARD_PREFIX}{run}/validation/{params['model']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']}/{params['weight_decay']}")
        training_writer = SummaryWriter(f"{TENSORBOARD_PREFIX}{run}/training/{params['model']}/{params['optimizer']}/{params['scheduler']}/{params['dropout']}/{params['weight_decay']}")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True)

    with TemporaryDirectory() as tempdir:
        # save original weights for resetting
        torch.save(model.state_dict(), os.path.join(tempdir, "original_weights.pt"))

        count_folds = 0
        for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
            # if fold > 0: continue
            count_folds += 1
            model.load_state_dict(
                torch.load(os.path.join(tempdir, "original_weights.pt"))
            )

            train_ds = Subset(train_dataset, train_indices)

            subset_data = [train_dataset[idx] for idx in train_ds.indices]
            _, subset_labels, _ = zip(*subset_data)
            labels = torch.tensor(subset_labels).long()
            train_counts = torch.bincount(labels)
            train_class_weights = len(labels) / (2.0 * train_counts.float())
            train_weights = train_class_weights[labels]
            train_sampler = WeightedRandomSampler(train_weights, len(labels))

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=batch_size, shuffle=False, sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, val_indices), batch_size=batch_size, shuffle=False
            )

            for epoch in range(NUM_EPOCHS):
                ### TRAINING ###

                model.train()
                # epoch_loss = 0.0
                # epoch_labels = torch.tensor([])
                # epoch_outputs = torch.tensor([])
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
                        class_weights = batch_size / (2.0 * class_counts.float())
                        sample_weights = class_weights[labels.long()]
                        criterion = nn.BCELoss(weight=sample_weights)

                    loss = criterion(outputs.flatten(), labels.float())
                    loss.backward()
                    optimizer.step()

                    if scheduler:
                        if type(scheduler) is lr_scheduler.ReduceLROnPlateau:
                            scheduler.step(loss.item())
                        else:
                            scheduler.step()

                #     epoch_loss += outputs.shape[0] * loss.item()
                #     epoch_labels = torch.cat(
                #         (epoch_labels, labels.float().detach().cpu()), 0
                #     )
                #     epoch_outputs = torch.cat(
                #         (epoch_outputs, outputs.flatten().detach().cpu()), 0
                #     )

                # epoch_loss = epoch_loss / len(epoch_labels)

                # if DEBUG and fold == 0:
                #     training_writer.add_scalar("loss", epoch_loss, epoch)
                #     training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
                    # training_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
                    # training_writer.add_scalar("precision", precision(epoch_outputs, epoch_labels), epoch)

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

                mean_f1[epoch] += f1(epoch_outputs, epoch_labels)
                mean_loss[epoch] += epoch_loss

                # if DEBUG and fold == 0:
                #     validation_writer.add_scalar("loss", epoch_loss, epoch)
                #     validation_writer.add_scalar("f1", f1_score, epoch)
                    # validation_writer.add_scalar("recall", recall(epoch_outputs, epoch_labels), epoch)
                    # validation_writer.add_scalar("precision", precision(epoch_outputs, epoch_labels), epoch)

    mean_loss = np.divide(mean_loss, count_folds)
    mean_f1 = np.divide(mean_f1, count_folds)

    if DEBUG:
        print(f"mean_loss: {mean_loss.shape} mean_f1: {mean_f1.shape}")
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


comm = MPI.COMM_WORLD
num_generations = 4
pop_size = 2 * MPI.COMM_WORLD.size
# limits = {
#     "model": ("resnet18", "resnet50", "efficientnet_s", "inception"),
#     "scheduler": ("step", "plateau", "none"),
#     "lr": (0.05, 0.0001),
#     "batch_size": (4, 256),
#     "optimizer": ("sgd_0", "sgd_09", "adam", "adagrad"),
#     "dropout": (0.0, 0.8),
#     "weight_decay": (0.0, 5.0),
# }
limits = {
    "model": ("resnet18", "resnet50"),
    "scheduler": ("step", "plateau", "none"),
    "loss": ("bce", "bce_weighted"),
    "lr": (0.05, 0.0001),
    "batch_size": (4, 256),
    "optimizer": ("sgd_0", "sgd_09", "adam", "adagrad"),
    "dropout": (0.0, 0.8),
    "weight_decay": (0.0, 5.0),
}
rng = random.Random(MPI.COMM_WORLD.rank)
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
propulator.summarize(top_n=10, debug=2)
