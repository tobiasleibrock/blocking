import torch
import time
from tempfile import TemporaryDirectory
import os
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold

training_writer = SummaryWriter("runs/training")    
validation_writer = SummaryWriter("runs/validation")    

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)

f1 = torchmetrics.F1()

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    train_features, train_labels = next(iter(dataloaders["train"]))
    writer.add_graph(model, train_features)

    since = time.time()
    
    model.to(device)
    
    metric = BinaryAccuracy(threshold=0.5).to(device)

    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"fold {fold+1}/{5}")

        best_acc = 0.0
        run_train = 0
        run_val = 0

        for epoch in range(num_epochs):
            print(f"epoch {epoch}/{num_epochs - 1}")

            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs.float())
                        preds = metric(outputs.flatten(), labels)
                        loss = criterion(outputs.flatten(), labels.float())

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += preds * len(labels)

                    # tensorboard
                    if phase == "train":
                        training_writer.add_scalar('loss', loss.item(), run_train)
                        training_writer.add_scalar('Accuracy/train', preds, run_train)
                        training_writer.add_scalar('f1', f1(outputs, labels), run_val)
                        run_train += 1
                    if phase == "val":
                        validation_writer.add_scalar('loss', loss.item(), run_val)
                        validation_writer.add_scalar('accuracy', preds, run_val)
                        validation_writer.add_scalar('f1', f1(outputs, labels), run_val)
                        run_val += 1
                
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print(f"{phase} loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

        time_elapsed = time.time() - since
        print(
            f"training in in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"best val Acc: {best_acc:4f}")
        
    return model
