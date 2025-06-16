from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import os
import torch
import importlib
import torch.optim as optim

from models.model_v1 import CNN
import torch.nn as nn

#Global constant declaration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCH_FOLDER_DIR = "epochs"
NUM_EPOCHS = 100

def preprocessing(isTraining: bool, folder: str) -> DataLoader:
    #image augmentation to improve the robustness of the model. This is done on the fly
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])

    validation_transforms = transforms.Compose([
    transforms.ToTensor()])

    #imageFolder expects the folder to be arranged in terms of the class labels
    dataset = datasets.ImageFolder(root=folder, transform=train_transforms)
    if (not isTraining) :
        dataset = datasets.ImageFolder(root=folder, transform=validation_transforms)

    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) #load data in parallel

def create_optimizer(model, optimizer_name='RMSprop', learning_rate=LEARNING_RATE, **kwargs):
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
  
    kwargs['lr'] = learning_rate
  
    optimizer = optimizer_class(model.parameters(), **kwargs)
    return optimizer
 
#dynamic loading of models
def load_model_class(file_name, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name)
    return model_class

def training(train_dataloaders: DataLoader, eval_dataloaders: DataLoader, epoch_folder_path: str, epoch_filepath: str,  num_epochs: int):
    #use hardware to accelerate the process if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    #model instantiation + loading in state dict if provided
    model = CNN().to(device)
    if (epoch_filepath):
        model.load_state_dict(torch.load(epoch_filepath), map_location=device) 

    #instantiating other variables
    optimizer = create_optimizer(model, "RMSprop", LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()

    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_eval_loss * 10000:.4f}')


if __name__ == "__main__":
    
    print("Processing and loading data...")
    #data preprocessing
    train_dataloaders = preprocessing(isTraining=True, folder="train/train")
    eval_dataloaders = preprocessing(isTraining=False, folder="train/validation")

    #actual training loop
    training(train_dataloaders=train_dataloaders, eval_dataloaders=eval_dataloaders, epoch_folder_path=EPOCH_FOLDER_DIR, epoch_filepath="",  num_epochs=NUM_EPOCHS)