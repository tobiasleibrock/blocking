import os
import time
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall

BATCH_SIZE = 64

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu" if torch.backends.mps.is_available() else "cpu"
)

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

current_datetime = datetime.now()
run = current_datetime.strftime("%Y-%m-%d %H:%M")


def log_day_set(writer, data):
    img_grid = torchvision.utils.make_grid(data.view((1, -1, 100)))
    writer.add_image("false_positives", img_grid)


def train_model(model, optimizer, scheduler, datasets, info, num_epochs=25):
    since = time.time()
    print("starting training..")

    model.to(device)
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    ukesm_dataset = datasets["ukesm"]

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    ukesm_loader = torch.utils.data.DataLoader(
        ukesm_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    kf = KFold(n_splits=5, shuffle=True)

    with TemporaryDirectory() as tempdir:

        # save original weights for resetting
        torch.save(model.state_dict(), os.path.join(tempdir, "original_weights.pt"))

        for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
            if fold != 0:
                continue
            print(f"fold {fold+1}/{5}")

            model.load_state_dict(
                torch.load(os.path.join(tempdir, "original_weights.pt"))
            )

            training_writer = SummaryWriter(f"runs/training/{run}/{fold}/{info}")
            validation_writer = SummaryWriter(f"runs/validation/{run}/{fold}/{info}")
            testing_writer = SummaryWriter(f"runs/test/{run}/{fold}/{info}")
            ukesm_writer = SummaryWriter(f"runs/ukesm/{run}/{fold}/{info}")

            train_ds = Subset(train_dataset, train_indices)

            subset_data = [train_dataset[idx] for idx in train_ds.indices]
            _, subset_labels = zip(*subset_data)
            labels = torch.tensor(subset_labels).long()
            train_counts = torch.bincount(labels)
            train_class_weights = len(labels) / (2.0 * train_counts.float())
            train_weights = train_class_weights[labels]
            train_sampler = WeightedRandomSampler(train_weights, len(labels))

            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler
            )
            val_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False
            )

            for epoch in range(num_epochs):
                print(f"epoch {epoch}/{num_epochs - 1}")

                ### TRAINING ###

                model.train()
                epoch_loss = 0.0
                epoch_labels = torch.tensor([])
                epoch_outputs = torch.tensor([])
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs.float())

                    # scale loss weights by class imbalance in input data
                    class_counts = torch.bincount(labels.long())
                    class_weights = 64 / (2.0 * class_counts.float())
                    sample_weights = class_weights[labels.long()]
                    criterion = nn.BCELoss(weight=sample_weights)
                    # criterion = nn.BCELoss()

                    loss = criterion(outputs.flatten(), labels.float())
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                    epoch_loss += outputs.shape[0] * loss.item()
                    epoch_labels = torch.cat(
                        (epoch_labels, labels.float().detach().cpu()), 0
                    )
                    epoch_outputs = torch.cat(
                        (epoch_outputs, outputs.flatten().detach().cpu()), 0
                    )

                epoch_loss = epoch_loss / len(epoch_labels)

                training_writer.add_scalar("loss", epoch_loss, epoch)
                training_writer.add_scalar(
                    "recall", recall(epoch_outputs, epoch_labels), epoch
                )
                training_writer.add_scalar(
                    "precision", precision(epoch_outputs, epoch_labels), epoch
                )
                training_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)

                ### VALIDATION ###

                model.eval()
                epoch_loss = 0.0
                epoch_labels = torch.tensor([])
                epoch_outputs = torch.tensor([])
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)

                        epoch_loss += outputs.shape[0] * loss.item()
                        epoch_labels = torch.cat(
                            (epoch_labels, labels.float().detach().cpu()), 0
                        )
                        epoch_outputs = torch.cat(
                            (epoch_outputs, outputs.flatten().detach().cpu()), 0
                        )

                        # img_grid = torchvision.utils.make_grid(inputs.view((64, 1, -1, 100)))
                        # validation_writer.add_image("val_geo", img_grid)

                epoch_loss = epoch_loss / len(epoch_labels)

                validation_writer.add_scalar("loss", epoch_loss, epoch)
                validation_writer.add_scalar(
                    "recall", recall(epoch_outputs, epoch_labels), epoch
                )
                validation_writer.add_scalar(
                    "precision", precision(epoch_outputs, epoch_labels), epoch
                )
                validation_writer.add_scalar(
                    "f1", f1(epoch_outputs, epoch_labels), epoch
                )

                # confusion matrix plotting
                conf_matrix = confusion_matrix(
                    epoch_labels, (epoch_outputs >= 0.5).int(), labels=np.array([0, 1])
                )
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=conf_matrix,
                    display_labels=["no blocking", "blocking"],
                )
                disp.plot()
                validation_writer.add_figure(
                    "conf-matrix", disp.figure_, global_step=epoch
                )

                ### TESTING (TEST) ###

                model.eval()
                epoch_loss = 0.0
                epoch_labels = torch.tensor([])
                epoch_outputs = torch.tensor([])
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)

                        epoch_loss += outputs.shape[0] * loss.item()
                        epoch_labels = torch.cat(
                            (epoch_labels, labels.float().detach().cpu()), 0
                        )
                        epoch_outputs = torch.cat(
                            (epoch_outputs, outputs.flatten().detach().cpu()), 0
                        )

                epoch_loss = epoch_loss / len(epoch_labels)

                testing_writer.add_scalar("loss", epoch_loss, epoch)
                testing_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
                testing_writer.add_scalar(
                    "recall", recall(epoch_outputs, epoch_labels), epoch
                )
                testing_writer.add_scalar(
                    "precision", precision(epoch_outputs, epoch_labels), epoch
                )

                ### TESTING (UKESM) ###

                # model.eval()
                # epoch_loss = 0.0
                # epoch_labels = torch.tensor([])
                # epoch_outputs = torch.tensor([])
                # with torch.no_grad():
                ##   for inputs, labels in ukesm_loader:
                #       inputs, labels = inputs.to(device), labels.to(device)
                #       outputs = model(inputs)
                #        _, predictions = torch.max(outputs, 1)
                #
                #        epoch_loss += outputs.shape[0] * loss.item()
                #        epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
                #        epoch_outputs = torch.cat((epoch_outputs, outputs.flatten().detach().cpu()), 0)
                #
                # epoch_loss = epoch_loss / len(epoch_labels)
            #
            # ukesm_writer.add_scalar('loss', epoch_loss, epoch)
            # ukesm_writer.add_scalar('f1', f1(epoch_outputs, epoch_labels), epoch)
            # ukesm_writer.add_scalar('recall', recall(epoch_outputs, epoch_labels), epoch)
            # ukesm_writer.add_scalar('precision', precision(epoch_outputs, epoch_labels), epoch)

            time_elapsed = time.time() - since
            print(
                f"training finished: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )

    return model
