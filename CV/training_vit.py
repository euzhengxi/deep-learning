import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.CustomCNN_v2 import CNN 
from models.MobileNet import MobileNet
from models.DeiT import DeiT
from preprocessing import preprocessing

#Global constants declaration
BATCH_SIZE = 32
LEARNING_RATE = 0.0001 #0.01
EPOCH_FOLDER_DIR = "epochs"
EPOCH_FILEPATH = f'best_epochs/deit_model_epoch_31.pt'
NUM_EPOCHS = 200

def create_optimizer(model, optimizer_name='RMSprop', learning_rate=LEARNING_RATE, **kwargs):
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
  
    kwargs['lr'] = learning_rate
    kwargs['weight_decay'] = 1e-4
  
    optimizer = optimizer_class(model.parameters(), **kwargs)
    return optimizer

def training(train_dataloader: DataLoader, eval_dataloader: DataLoader, epoch_folder_path: str, epoch_filepath: str,  num_epochs: int):
   
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")
    
    #use hardware acceleration if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    model = DeiT().to(device)
    if (epoch_filepath):
        model.load_state_dict(torch.load(EPOCH_FILEPATH, weights_only=True)) 
        
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.model.head.parameters():
        param.requires_grad = True

     #instantiating other variables
    optimizer = create_optimizer(model, "AdamW", LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        
        #training loop
        model.train()
        training_loss = 0.0
        num_batches = 0
        for batch in train_dataloader:
            inputs, labels = batch
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            pred = model(inputs)
            loss = loss_function(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            num_batches+=1
            training_loss += loss.item()

        avg_train_loss = training_loss / num_batches
        model_path = os.path.join(epoch_folder_path, f'model_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), model_path)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        
        #evaluation loop
        model.eval()
        eval_loss = 0.0
        num_batches = 0
        with torch.inference_mode():
            for batch in eval_dataloader:
                inputs, labels = batch
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(torch.float32).to(device)
                pred = model(inputs)
                
                loss = loss_function(pred, labels)
                num_batches+=1
                eval_loss += loss.item()
        
        avg_eval_loss = eval_loss / num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_eval_loss:.4f}')


if __name__ == "__main__":
    
    #data preprocessing
    print("\n>>> Processing and loading training data ...")
    train_dataloader = preprocessing(isTraining=True, isNewDataAdded=False, folder="train/training", batch_size=BATCH_SIZE)
    print(">>> Processing and loading validation data ...")
    eval_dataloader = preprocessing(isTraining=False, isNewDataAdded=False, folder="train/validation", batch_size=BATCH_SIZE)

    #actual training loop
    training(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, epoch_folder_path=EPOCH_FOLDER_DIR, epoch_filepath=EPOCH_FILEPATH,  num_epochs=NUM_EPOCHS)