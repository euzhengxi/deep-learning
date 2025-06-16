from PIL import Image 
import os 
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".heif", ".heic"]
ALLOWED_EXTENSIONS = (".jpg", ".png")

def get_num_pixels(filepath):
    width, height = Image.open(filepath).size
    return width, height


#getting the distribution of images in different folders
'''
with open("stats.txt", "a") as f:
    for folder in os.listdir("train"):
        dimensions = {}
        tdict = {}
        print(folder)
        for image in os.listdir(f'train/{folder}'):
            image_filepath = f'train/{folder}/{image}'
            width, height = get_num_pixels(image_filepath)
            filename, extension = os.path.splitext(image_filepath)
            dimensions[(width, height)] = dimensions[(width, height)] + 1 if (width, height) in dimensions else 1
            tdict[extension] = tdict[extension] + 1 if extension in tdict else 1
        
        f.write(f'\n\n{folder}\n\n')
        f.write(str(dimensions))
        f.write(f'\n')
        f.write(str(tdict))
        f.write(f'\n')
'''

#preliminary file processing to remove files except jpg / png, resize image to 128x128 and convert jpg to png
def process_images(dataset_dir: str):
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)

        if not os.path.isdir(folder_path):
            continue  # Skip files in the root "train" directory

        for image_name in os.listdir(folder_path):
            original_path = os.path.join(folder_path, image_name)
            filename, extension = os.path.splitext(original_path)
            extension = extension.lower()

            if (extension in IMAGE_EXTENSIONS):
                # Replace spaces in filenames
                if " " in image_name:
                    safe_name = image_name.replace(" ", "_")
                    safe_path = os.path.join(folder_path, safe_name)
                    os.rename(original_path, safe_path)
                    original_path = safe_path  # update path

                filename, extension = os.path.splitext(original_path)
                extension = extension.lower()

                # Remove files with invalid image extensions
                if extension not in ALLOWED_EXTENSIONS:
                    try:
                        os.remove(original_path)
                        print(f'{original_path} removed (invalid extension)')
                    except Exception as e:
                        print(f'Error removing {original_path}: {e}')
                    continue

                try:
                    img = Image.open(original_path).convert("RGB")
                    img = img.resize((128, 128))

                    # Save new image as .png
                    new_path = filename + ".png"
                    img.save(new_path, "PNG")

                    # Remove original file if it was .jpg or .jpeg
                    if extension in (".jpg", ".jpeg"):
                        os.remove(original_path)

                except Exception as e:
                    print(f'Error processing {original_path}: {e}')

#confirming changes made to the images
def get_stats(dataset_dir: str):
    with open("dataset_stats.txt", "a") as f:
        for folder in os.listdir(dataset_dir):
            dimensions = {}
            tdict = {}
            folder_path = os.path.join(dataset_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for image_name in os.listdir(folder_path):
                original_path = os.path.join(folder_path, image_name)
                width, height = get_num_pixels(original_path)
                filename, extension = os.path.splitext(original_path)
                dimensions[(width, height)] = dimensions[(width, height)] + 1 if (width, height) in dimensions else 1
                tdict[extension] = tdict[extension] + 1 if extension in tdict else 1
            
            f.write(f'\n\n{folder}\n')
            f.write(str(dimensions))
            f.write(f'\n')
            f.write(str(tdict))
            f.write(f'\n')

if __name__ == "__main__":
    train_dir = "train/train"
    eval_dir = "train/validation"

    print("Processing photos...")
    process_images(train_dir)
    process_images(eval_dir)

    print("Reading stats in train directory")
    get_stats(train_dir)
