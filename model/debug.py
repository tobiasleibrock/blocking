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
import albumentations as A
from torchvision.models.inception import Inception3

### CUSTOM MODULES ###
from resnet18 import get_model as get_resnet18_model
from resnet18 import get_model_10_channel as get_resnet18_10_channel_model
from resnet50 import get_model_10_channel as get_resnet50_10_channel_model
from efficientnet_s import get_model as get_efficientnet_model
from efficientnet_m import get_model as get_efficientnet_m_model
from inception_v3 import get_model as get_inception_model
from dataset import GeoEra5Dataset, GeoUkesmDataset, TransformDataset

INFO = "efficientnet_m-augmentation"
# BATCH_SIZE = 256
# LEARNING_RATE = 0.01
# DROPOUT = 0.0
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.0
# NUM_EPOCHS = 40
# NUM_YEARS = 10

BATCH_SIZE = 111
LEARNING_RATE = 0.004
DROPOUT = 0.437
MOMENTUM = 0.9
WEIGHT_DECAY = 0.132
NUM_EPOCHS = 20
NUM_YEARS = 10
DEBUG = False
TEST = True
TRANSFORM = True

device = torch.device("cpu")

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

transform = A.Compose(
    [
        A.GaussNoise(p=0.2),
        A.Rotate(limit=30, p=0.2),
        A.ChannelDropout(channel_drop_range=(1, 2), p=0.1),
    ]
)

era5_geo_dataset = GeoEra5Dataset()
# ukesm_dataset = GeoUkesmDataset()

# [{'model': 'inception', 'scheduler': 'plateau', 'loss': 'bce', 'sampler': 'weighted_random', 'augmentation': 'heavy', 'lr': '3.81E-3', 'batch_size': 111, 'optimizer': 'sgd_09', 'dropout': '4.37E-1', 'weight_decay': '1.32E-1'}, loss 2.29E-1, island 0, worker 3, generation 2]


def train_model(train_dataset, test_dataset, num_epochs):
    kf = KFold(n_splits=NUM_YEARS, shuffle=False)

    full_validation_labels = torch.tensor([])
    full_validation_outputs = torch.tensor([])

    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        print(f"fold {fold}/{NUM_YEARS}")
        if fold < 7:
            continue

        # model = get_resnet18_10_channel_model(dropout=DROPOUT)
        # model = get_resnet50_10_channel_model(dropout=DROPOUT)
        # model = get_resnet18_model(dropout=DROPOUT)
        model = get_inception_model(dropout=DROPOUT)
        # model = get_efficientnet_model(dropout=DROPOUT)
        # model = get_efficientnet_m_model(dropout=DROPOUT)

        model.to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        # optimizer = optim.Adagrad(
        #     model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        # )

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

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

        # train_ds = ConcatDataset([train_ds, ukesm_dataset])
        if TRANSFORM:
            train_ds = TransformDataset(subset_data, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
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

                # fix for inception model in pytorch
                # https://discuss.pytorch.org/t/inception-v3-is-not-working-very-well/38296/3
                if type(model) is Inception3:
                    outputs = model(inputs.float())[0]
                else:
                    outputs = model(inputs.float())

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
            # print(f"trn loss {epoch_loss}")
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
            # print((torch.bincount((epoch_outputs > 0.5).long()), torch.bincount(epoch_labels.long())))
            # print(f"val loss {epoch_loss}")
            print(f"val f1 {f1(epoch_outputs, epoch_labels)}")
            date_from = datetime.datetime(1900, 1, 1) + datetime.timedelta(
                hours=int(val_loader.dataset[0][2])
            )
            date_to = datetime.datetime(1900, 1, 1) + datetime.timedelta(
                hours=int(val_loader.dataset[-1][2])
            )
            print(f"validation from {date_from} to {date_to}")
            print("-----------------------------------")

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

            scheduler.step(loss.item())

    print("FINAL METRICS")
    print(INFO)
    print(f"f1: {f1(full_validation_outputs, full_validation_labels)}")
    print(f"recall: {recall(full_validation_outputs, full_validation_labels)}")
    print(f"precision: {precision(full_validation_outputs, full_validation_labels)}")

    return model


train_model(era5_geo_dataset, era5_geo_dataset, NUM_EPOCHS)
