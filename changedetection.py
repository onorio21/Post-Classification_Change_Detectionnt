import csv
import os
import torch
import numpy as np
import rasterio
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from PIL import Image
import re
from collections import defaultdict
from model import UNET
from utils import load_checkpoint

# Set up device
DEVICE = "cpu"

# Paths to the saved model and input/output files
model_checkpoint = '/Users/onorio21/Downloads/1model.pt'  # Update this path
prediction_image_folder = '/Users/onorio21/Desktop/Università/oldnew'   # Path where the images to predict are located
ground_truth_folder = '/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/GTMASK'    # Path where the ground truth images are located
output_folder = '/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/outchange'                 # Path where you want to save the results

# Threshold for binary classification
THRESHOLD = 0.5

# Load the checkpoint
checkpoint = torch.load(model_checkpoint)

# Load the model
model = UNET(in_channels=3, out_channels=2).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
# Move the model to the appropriate device
model.to(DEVICE)
#checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
#load_checkpoint(checkpoint, model)

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

#def save_prediction_as_png(prediction, output_path):
#    # Convert the prediction array to an image and save as PNG
#    image = Image.fromarray(prediction)
#    image.save(output_path, format="PNG")


def save_prediction_as_png(prediction, output_path):
    # If the prediction has more than one channel, select the first one
    if prediction.ndim == 3 and prediction.shape[0] == 2:
        prediction = prediction[0]  # Use the first channel

    # Convert the prediction array to an image and save as PNG
    image = Image.fromarray(prediction)
    image.save(output_path, format="PNG")

def get_location_from_filename(filename):
    # Extract location identifier, assuming the format "L15-1049E-1370N_4196_2710_13"
    match = re.search(r'(L\d{2}-\d{4}[A-Z]-\d{4}[A-Z]_\d{4}_\d{4}_\d{2})', filename)
    return match.group(1) if match else None

def load_image_as_array(image_path):
    image = Image.open(image_path)
    return np.array(image)

def evaluate_metrics(ground_truth, prediction):
    # Select the appropriate channel from prediction
    if prediction.ndim == 3 and prediction.shape[0] == 2:
        # Assuming the second channel represents the "change" class
        prediction = prediction[1]  # Use the second channel
    
    # Ensure both ground_truth and prediction are binary (0 or 1)
    ground_truth_binary = (ground_truth > 0).astype(int)
    prediction_binary = (prediction > 0).astype(int)

    # Check and print the shapes before flattening
    print(f"Ground truth shape: {ground_truth_binary.shape}")
    print(f"Prediction shape: {prediction_binary.shape}")

    # Flatten the arrays to compute the metrics
    ground_truth_flat = ground_truth_binary.flatten()
    prediction_flat = prediction_binary.flatten()

    # Ensure that the lengths match
    assert len(ground_truth_flat) == len(prediction_flat), "Mismatch in the number of samples between ground truth and prediction!"

    # Calculate metrics
    precision = precision_score(ground_truth_flat, prediction_flat)
    recall = recall_score(ground_truth_flat, prediction_flat)
    f1 = f1_score(ground_truth_flat, prediction_flat)
    accuracy = accuracy_score(ground_truth_flat, prediction_flat)
    
    return precision, recall, f1, accuracy

def main():

    print("creo i file")
    results_filename = 'metricschangesada.csv'

    with open(results_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Location', 'Precision', 'Recall', 'F1', 'Accuracy'])
    # Group images by location
    image_groups = defaultdict(list)
    
    # Get all image files in the prediction image folder
    for filename in os.listdir(prediction_image_folder):
        if filename.endswith('.tif'):
            location = get_location_from_filename(filename)
            if location:
                image_groups[location].append(filename)
    
    # Process each location
    for location, files in image_groups.items():
        if len(files) >= 2:
            metrics = []

            image_path_1 = os.path.join(prediction_image_folder, files[0])
            image_path_2 = os.path.join(prediction_image_folder, files[-1])

            # Predict the images
            prediction_1 = predict_image(image_path_1)
            prediction_2 = predict_image(image_path_2)

            # Perform change detection by comparing the two predictions
            change_detection_prediction = np.abs(prediction_2 - prediction_1)

            # Save the prediction with the specified naming convention
            prediction_filename = f"PRED_{location}.png"
            prediction_path = os.path.join(output_folder, prediction_filename)
            save_prediction_as_png(change_detection_prediction, prediction_path)

            # Load the corresponding ground truth image
            ground_truth_filename = f"MASK_{location}.png"
            ground_truth_path = os.path.join(ground_truth_folder, ground_truth_filename)
            
            if os.path.exists(ground_truth_path):
                ground_truth = load_image_as_array(ground_truth_path)

                # Evaluate the model's change detection against the ground truth
                precision, recall, f1, accuracy = evaluate_metrics(ground_truth, change_detection_prediction)

                metrics.append(precision)
                metrics.append(recall)
                metrics.append(f1)
                metrics.append(accuracy)
            
                
                
                with open(results_filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([location, metrics[0], metrics[1], metrics[2], metrics[3]])
                
                # Print the metrics
                print(f"Metrics for {location}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Accuracy: {accuracy:.4f}")
                print()
            else:
                print(f"Ground truth file not found for {ground_truth_filename}")

if __name__ == "__main__":
    main()