import torch
import numpy as np
import rasterio
import torch.nn.functional as F
from model import UNET
from utils import load_checkpoint
from PIL import Image

# Set up device
DEVICE = "cpu"

# Paths to the saved model and input/output files
model_checkpoint = "/Users/onorio21/Desktop/Università/buon train /best_model.pth.tar"
image_path_1 = "/Users/onorio21/Desktop/Università/train/L15-0331E-1257N_1327_3160_13/images_masked/global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160_13.tif"
image_path_2 = "/Users/onorio21/Desktop/Università/train/L15-0331E-1257N_1327_3160_13/images_masked/global_monthly_2020_01_mosaic_L15-0331E-1257N_1327_3160_13.tif"
prediction_1_path = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/out1.png"
prediction_2_path = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/out2.png"
output_change_detection_path = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/out3.png"

# Threshold for binary classification
THRESHOLD = 0.5

# Load the model
model = UNET(in_channels=3, out_channels=1).to(DEVICE)
checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
load_checkpoint(checkpoint, model)

def predict_image(image_path):
    # Load the image
    with rasterio.open(image_path) as src:
        # Read the first three bands (assuming image has 4 bands)
        image = src.read([1, 2, 3]).astype(np.float32)
    
    # Normalize image (assuming model was trained with images normalized to [0, 1])
    image = image / 255.0
    image = torch.tensor(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = torch.sigmoid(output).cpu().numpy().squeeze()

    # Binarize the prediction
    binary_prediction = (prediction > THRESHOLD).astype(np.uint8) * 255

    return binary_prediction

def save_prediction_as_png(prediction, output_path):
    # Convert the prediction array to an image and save as PNG
    image = Image.fromarray(prediction)
    image.save(output_path, format="PNG")

def main():
    # Predict the first and last month images
    prediction_1 = predict_image(image_path_1)
    prediction_2 = predict_image(image_path_2)

    # Save the predictions as PNG files
    save_prediction_as_png(prediction_1, prediction_1_path)
    save_prediction_as_png(prediction_2, prediction_2_path)

    # Perform change detection by comparing the two predictions
    change_detection = np.abs(prediction_1 - prediction_2)

    # Save the change detection result as a PNG file
    save_prediction_as_png(change_detection, output_change_detection_path)

if __name__ == "__main__":
    main()