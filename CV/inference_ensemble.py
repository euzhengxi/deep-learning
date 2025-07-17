import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

from models.CustomCNN_v2 import CNN 
from models.MobileNet import MobileNet
from models.DeiT import DeiT
from preprocessing import preprocessing

#Global constants declaration
BATCH_SIZE = 32
EPOCH_FOLDER_DIR = "best_epochs"
MODEL1_FILEPATH = f'{EPOCH_FOLDER_DIR}/deit_model_epoch_24.pt' 
MODEL2_FILEPATH = f'{EPOCH_FOLDER_DIR}/mobile_ce_model_epoch_198.pt' 

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

def inferencing(test_dataloader: DataLoader):
    if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
        print("Acceleration using MPS on Apple Silicon is not available")
    
    #use hardware to accelerate the process if it is available
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

    MODEL1 = DeiT().to(device)
    MODEL1.load_state_dict(torch.load(MODEL1_FILEPATH, weights_only=True)) 

    MODEL2 = MobileNet().to(device)
    MODEL2.load_state_dict(torch.load(MODEL2_FILEPATH, weights_only=True)) 

    #evaluation loop
    MODEL1.eval()
    MODEL2.eval()
    predictions = []
    targets = []
    class_weights = torch.tensor([0.9, #Anthracite
                                  0.9, #Conglomerate
                                  1, #Flint
                                  1, #Granite
                                  0, #Limestone
                                  0.9, #Marble
                                  0.8, #Nothing
                                  0.5, #Obsidian
                                  0.1, #Sandstone
                                  0.5 #Slate
                                  ])  
    class_weights = class_weights.unsqueeze(0)  # [1, num_classes]
    with torch.inference_mode():
        for inputs, labels in test_dataloader:
            inputs2 = inputs
            inputs1 = inputs.to(torch.float32).to(device)
            inputs2 = inputs2.to(torch.float32).to(device)
            logit1 = MODEL1(inputs1)
            logit2 = MODEL2(inputs2)

            class_weights = class_weights.to(logit1.device).view(1, -1)
            final_prob = class_weights * logit1 + (1 - class_weights) * logit2

            prediction = torch.softmax(final_prob, dim=1).argmax(1)
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


if __name__ == "__main__":
    
    print("\n>>> Processing and loading inferencing data...")
    #data preprocessing
    test_dataloader = preprocessing(isTraining=False, isNewDataAdded=True , folder="test", batch_size=BATCH_SIZE)

    #actual training loop
    inferencing(test_dataloader=test_dataloader)