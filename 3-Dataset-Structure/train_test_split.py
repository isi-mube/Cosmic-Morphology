import os
import shutil
from sklearn.model_selection import train_test_split

base_dir = "./nasa_images"
output_dir = "./3-Dataset-Structure"
classes = ["galaxy", "nebula", "planet", "star", "comet", "asteroid", "black hole"]
threshold = 47 # Maximum number of images per class

# Folder structure
for split in ["train", "val"]: # Iterate over train and val
    for cls in classes: # Iterate over each class
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True) # Create the folder structure

for cls in classes: # Iterate over each class
    class_path = os.path.join(base_dir, cls) # Path to the class folder
    if os.path.exists(class_path): #
        all_images = os.listdir(class_path) # Get all images
        limited_images = all_images[:threshold]  # Limit the number of images
        
        # Train-test split
        train_images, val_images = train_test_split(limited_images, test_size=0.3, random_state=42)

        # Copy to train folder
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, "train", cls, img)
            shutil.copy(src, dst)

        # Copy to val folder
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(output_dir, "val", cls, img)
            shutil.copy(src, dst)

print("Dataset structure created with a max of 47 images per class!")