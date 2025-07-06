import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from models.model_v2 import CNN 
from preprocessing import preprocessing

#Global constants declaration
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCH_FOLDER_DIR = "epochs"
EPOCH_FILEPATH = ""
NUM_EPOCHS = 200

def create_optimizer(model, optimizer_name='RMSprop', learning_rate=LEARNING_RATE, **kwargs):
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
  
    kwargs['lr'] = learning_rate
  
    optimizer = optimizer_class(model.parameters(), **kwargs)
    return optimizer

def training(train_dataloaders: DataLoader, eval_dataloaders: DataLoader, epoch_folder_path: str, epoch_filepath: str,  num_epochs: int):
   
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")
    
    #use hardware acceleration if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    #model instantiation + loading in state dict if provided
    model = CNN().to(device)
    if (epoch_filepath):
        model.load_state_dict(torch.load(epoch_filepath, weights_only=True)) 

    #instantiating other variables
    optimizer = create_optimizer(model, "RMSprop", LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        #training loop
        model.train()
        training_loss = 0.0
        num_batches = 0
        for batch in train_dataloaders:
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
        with torch.no_grad():
            for batch in eval_dataloaders:
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
    train_dataloaders = preprocessing(isTraining=True, isNewDataAdded=True , folder="train/training", batch_size=BATCH_SIZE)
    print(">>> Processing and loading validation data ...")
    eval_dataloaders = preprocessing(isTraining=False, isNewDataAdded=True , folder="train/validation", batch_size=BATCH_SIZE)

    #actual training loop
    training(train_dataloaders=train_dataloaders, eval_dataloaders=eval_dataloaders, epoch_folder_path=EPOCH_FOLDER_DIR, epoch_filepath=EPOCH_FILEPATH,  num_epochs=NUM_EPOCHS)