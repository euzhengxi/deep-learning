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
EPOCH_FOLDER_DIR = "epochs"

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


def inferencing(test_dataloader: DataLoader, epoch_filepath: str):
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")
    
    #use hardware to accelerate the process if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    model = CNN().to(device)
    if (epoch_filepath):
        model.load_state_dict(torch.load(epoch_filepath, weights_only=True)) 
    
    #evaluation loop
    model.eval()
    predictions = []
    targets = []    
    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(torch.float32).to(device)
            output = model(inputs)

            probabilities = nn.Softmax(dim=1)(output)
            prediction = probabilities.argmax(1)
            predictions.append(prediction)
            targets.append(labels.tolist())
    
    correct = 0
    for i in range(len(predictions)):
        predictionList = predictions[i]
        targetList = targets[i] 
        for i in range(len(predictionList)):
            print(f'actual: {pdict[targetList[i]]}, predicted: {pdict[predictionList[i].item()]}')   
            if (predictionList[i].item() == targetList[i]):
                correct += 1
    
    print(f'Accuracy: {correct}/{len(predictions) * len(predictions[0])}')
    return correct
    


if __name__ == "__main__":
    
    print("\n>>> Processing and loading inferencing data...")
    #data preprocessing
    test_dataloader = preprocessing(isTraining=False, isNewDataAdded=False , folder="test", batch_size=BATCH_SIZE)

    #actual inferencing loop
    epochs = [149, 190]
    for epoch in epochs:
        filepath = f'best_epochs/modelv2_epoch_{epoch}.pt'
        correct = inferencing(test_dataloader=test_dataloader, epoch_filepath=filepath)
        #print(f'{filepath} : {correct}')
        
