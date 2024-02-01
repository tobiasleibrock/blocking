import os
import time
from datetime import datetime, timedelta
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

### PLOTTING ###
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

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

def get_image(data, time):
    fig, axs = plt.subplots(nrows=5,ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()})

    axs = axs.flatten()
    clevs=np.arange(-5,5,1)
    long = np.arange(-45, 55, 1)
    lat = np.arange(30, 75, 1)

    for i in range(5):
        time = time + timedelta(days=1)
        axs[i].coastlines(resolution="110m",linewidth=1)
        cs = axs[i].contourf(long, lat, data[i].cpu(), clevs, transform=ccrs.PlateCarree(),cmap=plt.cm.jet)
        if i == 0: axs[i].set_title(str(time))

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(cs, cax=cbar_ax,orientation='vertical')

    plt.draw()

    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    fig_np = fig_np.transpose((2, 0, 1))

    plt.close(fig)

    return fig_np

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
            _, subset_labels, _ = zip(*subset_data)
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
                for inputs, labels, _ in train_loader:
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
                    for inputs, labels, t in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        predictions = (outputs > 0.5).float().flatten()
                        max_predictions = (outputs > 0.9).float().flatten()
                        min_predictions = (outputs > 0.2).float().flatten()
                        

                        epoch_loss += outputs.shape[0] * loss.item()
                        epoch_labels = torch.cat(
                            (epoch_labels, labels.float().detach().cpu()), 0
                        )
                        epoch_outputs = torch.cat(
                            (epoch_outputs, outputs.flatten().detach().cpu()), 0
                        )
        
                        if epoch == num_epochs - 1:
                            false_positives = (predictions == 1) & (labels == 0)
                            false_negatives = (predictions == 0) & (labels == 1)
                            #false_positives = (max_predictions == 1) & (labels == 0)
                            #false_negatives = (min_predictions == 0) & (labels == 1)

                            print("false_positives: " + str(torch.sum(false_positives).item()))
                            print("false_negatives: " + str(torch.sum(false_negatives).item()))
                        
                            for idx, (fp, fn) in enumerate(zip(false_positives, false_negatives)):
                                t_str = datetime(1900, 1, 1) + timedelta(hours=int(t[idx]))
                                if fp.item():
                                    img = get_image(inputs[idx], t_str)
                                    validation_writer.add_image(
                                        f"false-positive/{t_str.strftime('%Y-%m-%d')}", img, epoch
                                    )
                                if fn.item():
                                    img = get_image(inputs[idx], t_str)
                                    validation_writer.add_image(
                                        f"false-negative/{t_str.strftime('%Y-%m-%d')}", img, epoch
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

                # CONFUSION MATRIX
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

                # model.eval()
                # epoch_loss = 0.0
                # epoch_labels = torch.tensor([])
                # epoch_outputs = torch.tensor([])
                # with torch.no_grad():
                #     for inputs, labels in test_loader:
                #         inputs, labels = inputs.to(device), labels.to(device)
                #         outputs = model(inputs)
                #         _, predictions = torch.max(outputs, 1)

                #         epoch_loss += outputs.shape[0] * loss.item()
                #         epoch_labels = torch.cat(
                #             (epoch_labels, labels.float().detach().cpu()), 0
                #         )
                #         epoch_outputs = torch.cat(
                #             (epoch_outputs, outputs.flatten().detach().cpu()), 0
                #         )

                # epoch_loss = epoch_loss / len(epoch_labels)

                # testing_writer.add_scalar("loss", epoch_loss, epoch)
                # testing_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
                # testing_writer.add_scalar(
                #     "recall", recall(epoch_outputs, epoch_labels), epoch
                # )
                # testing_writer.add_scalar(
                #     "precision", precision(epoch_outputs, epoch_labels), epoch
                # )

                ### TESTING (UKESM) ###

            #     model.eval()
            #     epoch_loss = 0.0
            #     epoch_labels = torch.tensor([])
            #     epoch_outputs = torch.tensor([])
            #     with torch.no_grad():
            #         for inputs, labels in ukesm_loader:
            #             inputs, labels = inputs.to(device), labels.to(device)
            #             outputs = model(inputs)
            #             _, predictions = torch.max(outputs, 1)

            #             epoch_loss += outputs.shape[0] * loss.item()
            #             epoch_labels = torch.cat(
            #                 (epoch_labels, labels.float().detach().cpu()), 0
            #             )
            #             epoch_outputs = torch.cat(
            #                 (epoch_outputs, outputs.flatten().detach().cpu()), 0
            #             )

            #     epoch_loss = epoch_loss / len(epoch_labels)

            # ukesm_writer.add_scalar("loss", epoch_loss, epoch)
            # ukesm_writer.add_scalar("f1", f1(epoch_outputs, epoch_labels), epoch)
            # ukesm_writer.add_scalar(
            #     "recall", recall(epoch_outputs, epoch_labels), epoch
            # )
            # ukesm_writer.add_scalar(
            #     "precision", precision(epoch_outputs, epoch_labels), epoch
            # )

            time_elapsed = time.time() - since
            print(
                f"training finished: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )

    return model
