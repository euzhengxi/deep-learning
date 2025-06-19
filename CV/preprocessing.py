import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".heif", ".heic"]
ALLOWED_EXTENSIONS = (".jpg", ".png")

def get_num_pixels(filepath: str):
    width, height = Image.open(filepath).size
    return width, height

'''
Editing the images directly is intentional as it helps to lower the preprocessing time in subsequent training sessions
'''
def convert_and_resize_images(dataset_dir: str):
    print("Converting and resizing images ...")
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        # Skip files in the root directory
        if not os.path.isdir(folder_path):
            continue 

        for image_name in os.listdir(folder_path):
            #there shouldnt be any other files other images in the training / validation folders 
            #but the following code is structured to only modify image files - replace spaces in file name, resize and save it as jpg
            
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

    #image augmentation to improve the robustness of the model. This is done on the fly
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(128), #random scale and translation transformation, not just cropping
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ])

    standard_transforms = transforms.Compose([
    transforms.ToTensor()])

    #function expects the folder to be arranged in terms of the class labels
    dataset = datasets.ImageFolder(root=folder, transform=train_transforms)
    if (not isTraining) :
        dataset = datasets.ImageFolder(root=folder, transform=standard_transforms)
        return DataLoader(dataset, batch_size=batch_size, num_workers=4) 

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)