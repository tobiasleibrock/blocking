import torch
import time
from tempfile import TemporaryDirectory
import os
from torchmetrics.classification import BinaryF1Score, BinaryRecall, BinaryPrecision
from datetime import datetime
import torchmetrics
import torchvision
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

BATCH_SIZE = 64   

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)

f1 = BinaryF1Score(threshold=0.5).to(device)
recall = BinaryRecall(threshold=0.5).to(device)
precision = BinaryPrecision(threshold=0.5).to(device)

current_datetime = datetime.now()
run = current_datetime.strftime("%Y-%m-%d %H:%M")

def train_model(model, optimizer, scheduler, datasets, num_epochs=25):
    since = time.time()
    print("starting training..")
    
    model.to(device)
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    ukesm_dataset = datasets["ukesm"]
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    kf = KFold(n_splits=2, shuffle=True)

    with TemporaryDirectory() as tempdir:
        
        # save original weights for resetting
        torch.save(model.state_dict(), os.path.join(tempdir, "original_weights.pt"))
        
        for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
            print(f"fold {fold+1}/{5}")

            model.load_state_dict(torch.load(os.path.join(tempdir, "original_weights.pt")))
    
            training_writer = SummaryWriter(f"runs/training/{run}/{fold}")    
            validation_writer = SummaryWriter(f"runs/validation/{run}/{fold}") 
            testing_writer = SummaryWriter(f"runs/test/{run}/{fold}") 
    
            train_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=False
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
                    
                    loss = criterion(outputs.flatten(), labels.float())
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    
                    epoch_loss += outputs.shape[0] * loss.item()
                    epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
                    epoch_outputs = torch.cat((epoch_outputs, outputs.flatten().detach().cpu()), 0)

                epoch_loss = epoch_loss / len(epoch_labels)

                training_writer.add_scalar('loss', epoch_loss, epoch)
                #training_writer.add_scalar('recall', recall(epoch_outputs, epoch_labels), epoch)
                #training_writer.add_scalar('precision', precision(epoch_outputs, epoch_labels), epoch)
                training_writer.add_scalar('f1', f1(epoch_outputs, epoch_labels), epoch)

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
                        epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
                        epoch_outputs = torch.cat((epoch_outputs, outputs.flatten().detach().cpu()), 0)

                        #img_grid = torchvision.utils.make_grid(inputs.view((64, 1, -1, 100)))
                        #validation_writer.add_image("val_geo", img_grid)

                epoch_loss = epoch_loss / len(epoch_labels)
    
                validation_writer.add_scalar('loss', epoch_loss, epoch)
                validation_writer.add_scalar('recall', recall(epoch_outputs, epoch_labels), epoch)
                validation_writer.add_scalar('precision', precision(epoch_outputs, epoch_labels), epoch)
                validation_writer.add_scalar('f1', f1(epoch_outputs, epoch_labels), epoch)
    
                conf_matrix = confusion_matrix(epoch_labels, (epoch_outputs >= 0.5).int(), labels=np.array([0, 1]))
                
                fig, ax = plt.subplots()
                im = ax.imshow(conf_matrix, cmap='viridis')
                plt.colorbar(im)
                validation_writer.add_figure('conf-matrix', fig, global_step=epoch)

                ### TESTING (UKESM) ###

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
                        epoch_labels = torch.cat((epoch_labels, labels.float().detach().cpu()), 0)
                        epoch_outputs = torch.cat((epoch_outputs, outputs.flatten().detach().cpu()), 0)

                epoch_loss = epoch_loss / len(epoch_labels)
                
                testing_writer.add_scalar('loss', epoch_loss, epoch)
                testing_writer.add_scalar('f1', f1(epoch_outputs, epoch_labels), epoch)
            
            time_elapsed = time.time() - since
            print(
                f"training finished: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        
    return model
