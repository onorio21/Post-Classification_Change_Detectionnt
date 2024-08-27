import os
import torch
import rasterio

# Initialize counts
num_positives = 0
num_negatives = 0

# Path to your main labels folder
main_folder = "/Users/onorio21/Desktop/Università/conversions"

# Traverse through all subfolders and process .tif files
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith(".tif"):
            mask_path = os.path.join(root, file)

            # Load the mask
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Assuming mask is stored in the first channel

            # Count positive and negative pixels
            num_positives += (mask == 255).sum()
            num_negatives += (mask == 0).sum()

# Print the counts
print(f"Number of positive pixels: {num_positives}")
print(f"Number of negative pixels: {num_negatives}")

# Calculate pos_weight
if num_positives == 0:
    print("No positive pixels found in the dataset.")
else:
    pos_weight = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to('mps')
    print(f"Calculated pos_weight: {pos_weight.item()}")