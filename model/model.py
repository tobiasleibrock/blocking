### ML ###
import datetime
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, WeightedRandomSampler, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall
from sklearn.model_selection import KFold

### PLOTTING ###

### CUSTOM MODULES ###
from resnet18 import get_model as get_resnet18_model
from resnet18 import get_model_10_channel as get_resnet18_10_channel_model
from resnet50 import get_model_10_channel as get_resnet50_10_channel_model
from efficientnet_s import get_model as get_efficientnet_model
from inception_v3 import get_model as get_inception_model
from dataset import BlockingObservationalDataset1x1, BlockingUKESMDataset1x1, GeopotentialEra5UkesmDataset, GeopotentialSlpEra5Dataset, SlpObservationalDataset

INFO = "inception"
BATCH_SIZE = 256
LEARNING_RATE = 0.01
DROPOUT = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 0.1
NUM_EPOCHS = 15
NUM_YEARS = 10
DEBUG = False
TEST = True

device = torch.device("cuda:0")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

era5_geo_dataset = BlockingObservationalDataset1x1()
era5_slp_dataset = SlpObservationalDataset()
era5_geo_slp_dataset = GeopotentialSlpEra5Dataset()
ukesm_dataset = BlockingUKESMDataset1x1()
era5_ukesm_geo = GeopotentialEra5UkesmDataset()


def train_model(train_dataset, test_dataset, num_epochs):
    kf = KFold(n_splits=NUM_YEARS, shuffle=False)

    full_validation_labels = torch.tensor([])
    full_validation_outputs = torch.tensor([])

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        print(f"fold {fold}/{NUM_YEARS}")

        # model = get_resnet18_10_channel_model(dropout=DROPOUT)
        # model = get_resnet50_10_channel_model(dropout=DROPOUT)
        # model = get_resnet18_model(dropout=DROPOUT)
        model = get_inception_model(dropout=DROPOUT)

        model.to(device)
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        if DEBUG:
            validation_writer = SummaryWriter(f"runs/{INFO}/validation/{fold}")
            testing_writer = SummaryWriter(f"runs/{INFO}/test/{fold}")
            training_writer = SummaryWriter(f"runs/{INFO}/training/{fold}")

        train_ds = Subset(train_dataset, train_indices)

        subset_data = [train_dataset[idx] for idx in train_ds.indices]
        _, subset_labels, _ = zip(*subset_data)
        labels = torch.tensor(subset_labels).long()
        train_counts = torch.bincount(labels)
        train_class_weights = len(labels) / (2.0 * train_counts.float())
        train_weights = train_class_weights[labels]
        train_sampler = WeightedRandomSampler(train_weights, len(labels))

        train_ds = ConcatDataset([train_ds, ukesm_dataset])

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            Subset(train_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False
        )

        for epoch in range(num_epochs):
            ### TRAINING ###
            model.train()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            for inputs, labels, time in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs.float())[0]

                class_counts = torch.bincount(labels.long())
                class_weights = BATCH_SIZE / (2.0 * class_counts.float())
                sample_weights = class_weights[labels.long()]
                criterion = nn.BCELoss(weight=sample_weights)
                # criterion = nn.BCELoss()

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

            ### VALIDATION ###
            model.eval()
            epoch_loss = 0.0
            epoch_labels = torch.tensor([])
            epoch_outputs = torch.tensor([])
            with torch.no_grad():
                for inputs, labels, time in val_loader:
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
            print((torch.bincount((epoch_outputs > 0.5).long()), torch.bincount(epoch_labels.long())))
            print(f"val f1 {f1(epoch_outputs, epoch_labels)}")

            if DEBUG:
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

            if epoch == num_epochs - 1:
                full_validation_labels = torch.cat(
                    (full_validation_labels, epoch_labels), 0
                )
                full_validation_outputs = torch.cat(
                    (full_validation_outputs, epoch_outputs), 0
                )

            # CONFUSION MATRIX
            # if DEBUG:
            #     conf_matrix = confusion_matrix(
            #         epoch_labels,
            #         (epoch_outputs >= 0.5).int(),
            #         labels=np.array([0, 1]),
            #     )
            #     disp = ConfusionMatrixDisplay(
            #         confusion_matrix=conf_matrix,
            #         display_labels=["no blocking", "blocking"],
            #     )
            #     disp.plot()
            #     validation_writer.add_figure(
            #         "conf-matrix", disp.figure_, global_step=epoch
            #     )

            ### TESTING ###
            # if TEST:
            #     model.eval()
            #     epoch_loss = 0.0
            #     epoch_labels = torch.tensor([])
            #     epoch_outputs = torch.tensor([])
            #     with torch.no_grad():
            #         for inputs, labels in test_loader:
            #             inputs, labels = inputs.to(device), labels.to(device)
            #             outputs = model(inputs)

            #             epoch_loss += outputs.shape[0] * loss.item()
            #             epoch_labels = torch.cat(
            #                 (epoch_labels, labels.float().detach().cpu()), 0
            #             )
            #             epoch_outputs = torch.cat(
            #                 (epoch_outputs, outputs.flatten().detach().cpu()), 0
            #             )

            #     epoch_loss = epoch_loss / len(epoch_labels)

            #     if DEBUG:
            #         testing_writer.add_scalar("loss", epoch_loss, epoch)
            #         testing_writer.add_scalar(
            #             "f1", f1(epoch_outputs, epoch_labels), epoch
            #         )
            #         testing_writer.add_scalar(
            #             "recall", recall(epoch_outputs, epoch_labels), epoch
            #         )
            #         testing_writer.add_scalar(
            #             "precision", precision(epoch_outputs, epoch_labels), epoch
            #         )

            scheduler.step()

    print("FINAL METRICS")
    print(INFO)
    print(f"f1: {f1(full_validation_outputs, full_validation_labels)}")
    print(f"recall: {recall(full_validation_outputs, full_validation_labels)}")
    print(f"precision: {precision(full_validation_outputs, full_validation_labels)}")

    return model


train_model(era5_geo_dataset, era5_geo_dataset, NUM_EPOCHS)
