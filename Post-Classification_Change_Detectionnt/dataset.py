import os
import rasterio
from torch.utils.data import Dataset
import numpy as np

class SN7BuildingsDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform
        with open(txt_file, 'r') as f:
            self.image_label_pairs = [line.strip().split() for line in f]

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, index):
        img_path, mask_path = self.image_label_pairs[index]

        # Load the image with rasterio
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3]).astype(np.float32)  # Read the first 3 channels (RGB)

        # Load the mask with rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # Read only the first channel

        # Ensure the mask is binary (0 and 1)
        mask[mask == 255.0] = 1.0

        # If sizes do not match, raise an error
        if image.shape[1:] != mask.shape:
            raise ValueError(f"Image and mask sizes do not match: {image.shape[1:]} vs {mask.shape}")

        if self.transform is not None:
            # Moveaxis to convert image from (C, H, W) to (H, W, C) before passing to albumentations
            image = np.moveaxis(image, 0, -1)
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# Esempio di utilizzo
# dataset = SegmentationDataset(txt_file='/path/to/train.txt', transform=your_transform_function)