import numpy as np
import os
import psutil
import random
import torch
from PIL import Image
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


#Variables used in dataloader / dataset are declared at the top level to ensure pickability. 

def custom_worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".heif", ".heic"]

'''
Transforms are image augmentation to improve the robustness of the model. These transformations are done on the fly. 
1. randomresizedcrop - random scale and translation transformation, not just cropping
2. totensor - converts to tensor and normalises the values
'''
TRAIN_TRANSFORMS = v2.Compose([
    v2.RandomResizedCrop(128), 
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(15),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.RandomGrayscale(p=0.1),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

STANDARD_TRANSFORMS = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

def get_num_pixels(filepath: str):
    width, height = Image.open(filepath).size
    return width, height


#Editing the images directly is intentional as it helps to lower the preprocessing time in subsequent training sessions
def convert_and_resize_images(dataset_dir: str):
    print("Converting and resizing images ...")

    unknown_files = []
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        # Skip files in the root directory
        if not os.path.isdir(folder_path):
            continue 
        
        for image_name in os.listdir(folder_path):
            '''
            there shouldnt be any other files other images in the training / validation folders 
            but the following code is structured to only modify image files - replace spaces in file name, resize and save it as jpg
            '''
            
            
            filepath = os.path.join(folder_path, image_name)
            filename, extension = os.path.splitext(filepath)
            extension = extension.lower()

            if (extension in IMAGE_EXTENSIONS):
                # Replace spaces in filenames
                if " " in image_name:
                    safe_name = image_name.replace(" ", "_")
                    safe_path = os.path.join(folder_path, safe_name)
                    os.rename(filepath, safe_path)
                    filepath = safe_path  

                filename, extension = os.path.splitext(filepath)
                extension = extension.lower()

                #resizing and converting into png
                try:
                    img = Image.open(filepath).convert("RGB")
                    img = img.resize((128, 128))

                    # Save new image as .png
                    new_path = filename + ".jpg"
                    img.save(new_path, "JPEG")

                    if extension != ".jpg":
                        os.remove(filepath)

                except Exception as e:
                    print(f'Error processing {filepath}: {e}')
            else:
                unknown_files.append(filepath)
            
    if len(unknown_files) != 0:
        print(f'\nThe following files have unknown file extension:')
                
        for i in range(len(unknown_files)):
            print(f'{i + 1}. {unknown_files[i]}')
        print(f'Allowed extensions: {IMAGE_EXTENSIONS}. \nIf this is a valid image, pls edit convert_and_resize function in the preprocessing.py file \n')

def check_filetype_and_get_stats(dataset_dir: str):

    print("Performing checks on filetype and tabulating image distribution ... \n")
    imageData = {}
    
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.isdir(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            original_path = os.path.join(folder_path, image_name)
            filename, extension = os.path.splitext(original_path)
            if extension in IMAGE_EXTENSIONS:
                width, height = get_num_pixels(original_path)
                if extension != ".jpg":
                    raise Exception("Multiple image types detected")
                elif width != 128 or height != 128:
                    raise Exception("Multiple image dimensions detected")
                
                imageData[folder] = imageData[folder] + 1 if folder in imageData else 1

    print("Detected Images: ")
    for key in imageData:
        print(f'{key}: {imageData[key]}')
    print()

def preprocessing(isTraining: bool, isNewDataAdded:bool, folder: str, batch_size: int) -> DataLoader:
    if (isNewDataAdded):
        convert_and_resize_images(folder)
        check_filetype_and_get_stats(folder)

    memory = psutil.virtual_memory()
    memoryAvailable = memory.available / memory.total * 100
    if (memoryAvailable > 60 and torch.cuda.is_available()):
        print(f'\nAvailable memory: {memoryAvailable:.2f} %. CUDA detected, consider setting pin_memory=true in dataloader in preprocessing.py to accelerate data transfer\n')
    #function expects the folder to be arranged in terms of the class labels
    dataset = datasets.ImageFolder(root=folder, transform=TRAIN_TRANSFORMS)
    if (not isTraining) :
        dataset = datasets.ImageFolder(root=folder, transform=STANDARD_TRANSFORMS)
        return DataLoader(dataset, batch_size=batch_size, num_workers=4, worker_init_fn=custom_worker_init_fn) #, pin_memory=torch.cuda.is_available())

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=custom_worker_init_fn) #, pin_memory=torch.cuda.is_available())