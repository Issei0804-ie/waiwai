import torch
import glob
import albumentations as A
from PIL import Image
import numpy as np
import os
import argparse

AUGMENTATION_DIR = "aug_dataset"


def horizontalFlip(dataset_dir :str):
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])

    file_pathes = glob.glob(os.path.join(dataset_dir, "**", "*.jpg"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "horizontal_flip")
    os.makedirs(augmented_dir, exist_ok=True)

    for file_path in file_pathes:
        basename_without_ext = os.path.basename(file_path)
        augmented_filepath = os.path.join(augmented_dir, os.path.dirname(file_path), basename_without_ext)
        os.makedirs(os.path.dirname(augmented_filepath), exist_ok=True)
        img = Image.open(file_path)
        numpy_image = np.array(img)
        transformed = transform(image=numpy_image)
        transformed = Image.fromarray(transformed["image"])
        transformed.save(augmented_filepath)


def split_augdata():
    pass

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="augmentationを行うためのツール")
    p.add_argument("dataset_dir")
    args = p.parse_args()
    dataset_dir = args.dataset_dir
    horizontalFlip(dataset_dir)