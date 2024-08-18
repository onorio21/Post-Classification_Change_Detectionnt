import torch
import rasterio

# Initialize counts
num_positives = 0
num_negatives = 0

# Path to your dataset text file (assuming it's a list of image and mask paths)
dataset_txt_file = "/Users/onorio21/Desktop/train.txt"

# Read the paths to images and masks
with open(dataset_txt_file, 'r') as f:
    image_label_pairs = [line.strip().split() for line in f]

# Iterate through the dataset to calculate the number of positive and negative pixels
for img_path, mask_path in image_label_pairs:
    # Load the mask
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Assuming mask is stored in the first channel

    # Count positive and negative pixels
    num_positives += (mask == 255).sum()
    num_negatives += (mask == 0).sum()

# Print the counts
print(f"Number of positive pixels: {num_positives}")
print(f"Number of negative pixels: {num_negatives}")

# You can now use these counts to calculate pos_weight
pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to('mps')
print(f"Calculated pos_weight: {pos_weight.item()}")