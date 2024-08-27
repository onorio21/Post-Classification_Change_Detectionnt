import torch
import torchvision
import rasterio
import numpy as np
import os
from model import UNET  # Assuming your UNET model is defined in model.py
from utils import load_checkpoint, set_seed
from torchvision.transforms import functional as TF
from rasterio.transform import from_origin

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/best_model.pth.tar"  # Replace with your actual checkpoint path
IMG_PATH1 = "/Users/onorio21/Desktop/Università/train/L15-1025E-1366N_4102_2726_13/images_masked/global_monthly_2018_01_mosaic_L15-1025E-1366N_4102_2726_13.tif"  # Replace with the path to your first image
IMG_PATH2 = "/Users/onorio21/Desktop/Università/train/L15-1025E-1366N_4102_2726_13/images_masked/global_monthly_2020_01_mosaic_L15-1025E-1366N_4102_2726_13.tif"  # Replace with the path to your second image
OUTPUT_DIR = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/outchange"  # Replace with the desired output directory path
SEED = 42



def predict_change(model, img_path1, img_path2, output_dir):
    set_seed(SEED)
    
    # Load the images with rasterio
    with rasterio.open(img_path1) as src:
        image1 = src.read([1, 2, 3]).astype(np.float32)
    
    with rasterio.open(img_path2) as src:
        image2 = src.read([1, 2, 3]).astype(np.float32)
    
    # Preprocess images
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)
    
    # Predict masks for both images
    with torch.no_grad():
        mask1 = torch.sigmoid(model(image1)).cpu()
        mask2 = torch.sigmoid(model(image2)).cpu()
    
    # Apply threshold to make predictions binary
    mask1 = (mask1 > 0.5).float()
    mask2 = (mask2 > 0.5).float()
    
    # Compute the absolute difference between the two masks
    change_mask = torch.abs(mask1 - mask2)
    
    # Save the masks and change map
    save_predictions_and_change_map(mask1, mask2, change_mask, img_path1, output_dir)

def preprocess_image(image):
    # Normalize image and convert to torch tensor
    image = np.moveaxis(image, 0, -1)  # Convert (C, H, W) to (H, W, C)
    image = image / 255.0  # Scale to [0, 1]
    image = TF.to_tensor(image).unsqueeze(0).to(DEVICE)  # Add batch dimension
    return image

def save_predictions_and_change_map(mask1, mask2, change_map, reference_image_path, output_dir):
    # Load reference image to get the geo-transform and CRS
    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        crs = src.crs

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the prediction masks and change map as TIFF files
    mask1_np = mask1.squeeze().numpy()  # Remove batch and channel dimensions
    mask2_np = mask2.squeeze().numpy()  # Remove batch and channel dimensions
    change_map_np = change_map.squeeze().numpy()  # Remove batch and channel dimensions

    output_path_mask1 = os.path.join(output_dir, "prediction1.tif")
    output_path_mask2 = os.path.join(output_dir, "prediction2.tif")
    output_path_change_map = os.path.join(output_dir, "change_map.tif")

    with rasterio.open(
        output_path_mask1,
        'w',
        driver='GTiff',
        height=mask1_np.shape[0],
        width=mask1_np.shape[1],
        count=1,
        dtype=mask1_np.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask1_np, 1)

    with rasterio.open(
        output_path_mask2,
        'w',
        driver='GTiff',
        height=mask2_np.shape[0],
        width=mask2_np.shape[1],
        count=1,
        dtype=mask2_np.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask2_np, 1)

    with rasterio.open(
        output_path_change_map,
        'w',
        driver='GTiff',
        height=change_map_np.shape[0],
        width=change_map_np.shape[1],
        count=1,
        dtype=change_map_np.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(change_map_np, 1)

def main():
    # Load the model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(CHECKPOINT_PATH, map_location=DEVICE), model)
    model.eval()
    
    # Predict and save the change map
    predict_change(model, IMG_PATH1, IMG_PATH2, OUTPUT_DIR)

if __name__ == "__main__":
    main()