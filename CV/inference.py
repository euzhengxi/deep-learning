import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model_v2 import CNN 
from preprocessing import preprocessing

#Global constants declaration
BATCH_SIZE = 64
EPOCH_FOLDER_DIR = "epochs"
EPOCH_FILEPATH = "modelv2_epoch_149.pt"

pdict = {
    0: "Anthracite", 
    1: "Conglomerate", 
    2: "Flint", 
    3: "Granite", 
    4: "Limestone", 
    5: "Marble", 
    6: "Nothing",
    7: "Obsidian",  
    8: "Sandstone", 
    9: "Slate"}

actual_labels = []


def inferencing(test_dataloaders: DataLoader, epoch_filepath: str):
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")
    
    #use hardware to accelerate the process if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    #model instantiation + loading in state dict if provided
    model = CNN().to(device)
    model.load_state_dict(torch.load(epoch_filepath, weights_only=True)) 

    #evaluation loop
    model.eval()
    pred = []    
    with torch.no_grad():
        for batch in test_dataloaders:
            inputs, labels = batch
            inputs = inputs.to(torch.float32).to(device)
            actual_labels = labels.tolist()
            pred = model(inputs)
    
    prediction_classes = torch.argmax(pred, dim=1)
    correct = 0
    for i in range(len(prediction_classes)):
        print(f'actual: {pdict[actual_labels[i]]}, predicted: {pdict[prediction_classes[i].item()]}')   
        if (prediction_classes[i].item() == actual_labels[i]):
            correct += 1
    
    print(f'{correct}/{len(prediction_classes)}')
    


if __name__ == "__main__":
    
    print("Processing and loading inferencing data...")
    #data preprocessing
    test_dataloaders = preprocessing(isTraining=False, isNewDataAdded=True , folder="test", batch_size=BATCH_SIZE)

    #actual training loop
    inferencing(test_dataloaders=test_dataloaders, epoch_filepath=EPOCH_FILEPATH)