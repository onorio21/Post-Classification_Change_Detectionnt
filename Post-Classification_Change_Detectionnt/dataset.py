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

        # Carica l'immagine con rasterio
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)

        # Carica la maschera con rasterio
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # Legge solo il primo canale

        # Assicurati che la maschera sia binaria (0 e 1)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

# Esempio di utilizzo
# dataset = SegmentationDataset(txt_file='/path/to/train.txt', transform=your_transform_function)