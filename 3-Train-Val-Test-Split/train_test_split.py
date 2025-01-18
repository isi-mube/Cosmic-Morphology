import os
import shutil
from sklearn.model_selection import train_test_split
from random import shuffle
from PIL import Image

# Paths and class configuration
base_dir = "../1-Data-Collection/nasa_images"
output_dir = "."  # Current directory
classes = ["galaxy", "nebula", "planet", "star", "comet", "asteroid"]  # Exclude 'black_hole'
threshold = 500  # Maximum number of images per class
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}  # Supported extensions

def is_valid_image(file_path):
    """
    Check if the file is a valid image.

    Parameters:
    -----------
    file_path : str
        Path to the image file.

    Returns:
    --------
    bool
        True if the image is valid, False otherwise.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that the image can be opened
        return True
    except Exception:
        return False

# Create folder structure
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split and copy images
for cls in classes:
    class_path = os.path.join(base_dir, cls)
    if os.path.exists(class_path):
        # Filter valid images by extension and validate them
        all_images = [
            img for img in os.listdir(class_path)
            if os.path.splitext(img)[1].lower() in valid_extensions and
            is_valid_image(os.path.join(class_path, img))
        ]

        shuffle(all_images)  # Randomize the order
        limited_images = all_images[:min(len(all_images), threshold)]

        # Train-test split
        train_images, temp_images = train_test_split(limited_images, test_size=0.4, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

        # Log split sizes
        print(f"Class '{cls}': Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        # Copy to respective folders
        for split, images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
            for img in images:
                src = os.path.join(class_path, img)
                dst = os.path.join(output_dir, split, cls, img)
                if os.path.isfile(src):  # Check if file exists
                    shutil.copy(src, dst)
                else:
                    print(f"Warning: {src} is not a valid file!")
    else:
        print(f"Warning: Class folder '{cls}' does not exist at {class_path}.")

print(f"Dataset structure created with a max of {threshold} images per class!")