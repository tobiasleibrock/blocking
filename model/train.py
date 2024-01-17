import torch
import time
from tempfile import TemporaryDirectory
import os
from torchmetrics.classification import BinaryF1Score
from datetime import datetime
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import Subset

BATCH_SIZE = 64   

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    if torch.backends.mps.is_available()
    else "cpu"
)

f1 = BinaryF1Score(threshold=0.5).to(device)

current_datetime = datetime.now()
run = current_datetime.strftime("%Y-%m-%d %H:%M")

def train_model(model, criterion, optimizer, scheduler, datasets, num_epochs=25):
    since = time.time()
    print("starting training..")
    
    model.to(device)
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    kf = KFold(n_splits=5, shuffle=True)

    with TemporaryDirectory() as tempdir:
        
        # save original weights for resetting
        torch.save(model.state_dict(), os.path.join(tempdir, "original_weights.pt"))
        
        for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
            print(f"fold {fold+1}/{5}")

            model.load_state_dict(torch.load(os.path.join(tempdir, "original_weights.pt")))
    
            training_writer = SummaryWriter(f"runs/training/{run}/{fold}")    
            validation_writer = SummaryWriter(f"runs/validation/{run}/{fold}") 
    
            train_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=False
            )
            val_loader = torch.utils.data.DataLoader(
                Subset(train_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False
            )
    
            run_train = 0
            run_val = 0
    
            for epoch in range(num_epochs):
                print(f"epoch {epoch}/{num_epochs - 1}")
    
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs.float())
                    loss = criterion(outputs.flatten(), labels.float())
                    loss.backward()
                    optimizer.step()
                    #scheduler.step()
                    training_writer.add_scalar('loss', loss.item(), run_train)
                    #training_writer.add_scalar('accuracy', preds.float(), run_train)
                    training_writer.add_scalar('f1', f1(outputs.flatten(), labels.float()), run_train)
                    run_train += 1
    
                model.eval()
                with torch.no_grad():
                    total_correct = 0
                    total_samples = 0
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predictions = torch.max(outputs, 1)
                        validation_writer.add_scalar('loss', loss.item(), run_val)
                        #validation_writer.add_scalar('accuracy', preds.float(), run_val)
                        validation_writer.add_scalar('f1', f1(outputs.flatten(), labels.float()), run_val)
                        run_val += 1
    
            time_elapsed = time.time() - since
            print(
                f"training finished: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
            )
        
    return model
