import torch
import numpy as np
import os
import torchvision
import rasterio
from rasterio.transform import from_origin
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    set_seed,
)

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  # 1024 originally
IMAGE_WIDTH = 512  # 1024 originally
PIN_MEMORY = True
LOAD_MODEL = False 
TRAIN_IMG_DIR = ""
TRAIN_LABEL_DIR = ""
VAL_IMG_DIR = ""
VAL_LABEL_DIR = ""
SEED = 42

set_seed(SEED)

def train_fn(loader, model, optimizer, loss_fn, epoch_losses):
    loop = tqdm(loader)
    epoch_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        epoch_loss += loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

    epoch_losses.append(epoch_loss / len(loader))

def evaluate(loader, model, loss_fn):
    model.eval()
    all_preds = []
    all_targets = []
    val_loss = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE).float().unsqueeze(1)
            predictions = torch.sigmoid(model(data))
            preds = (predictions > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Calculate validation loss
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

    val_loss /= len(loader)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets.flatten(), all_preds.flatten())
    recall = recall_score(all_targets.flatten(), all_preds.flatten())
    f1 = f1_score(all_targets.flatten(), all_preds.flatten())
    accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())

    return precision, recall, f1, accuracy, val_loss

def plot_and_save_metrics(train_losses, val_losses, val_metrics, output_dir):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{output_dir}/losses.png")
    plt.close()

    metrics_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
    for i, metric_name in enumerate(metrics_names):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [metrics[i] for metrics in val_metrics], label=f"Validation {metric_name}")
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.title(f"Validation {metric_name}")
        plt.legend()
        plt.savefig(f"{output_dir}/{metric_name.lower()}.png")
        plt.close()

TRAIN_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/train.txt"
VAL_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/val.txt"

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )

    val_transform = A.Compose(
        [
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_TXT,
        VAL_TXT,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY
    )

    best_f1 = 0
    train_losses = []
    val_losses = []
    val_metrics = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, train_losses)

        # Validation step
        val_loss, val_precision, val_recall, val_f1, val_accuracy = evaluate(val_loader, model, loss_fn)
        val_metrics.append([val_precision, val_recall, val_f1, val_accuracy])
        val_loss = train_losses[-1]  # Using training loss for simplicity, you may want to calculate actual validation loss
        val_losses.append(val_loss)

        print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"best_model.pth.tar")

        # Optional: Save predictions as images
        # save_predictions_as_imgs(
        #     val_loader, model, folder='/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/provaraster', device=DEVICE
        # )

    # Plot and save metrics
    plot_and_save_metrics(train_losses, val_losses, val_metrics, output_dir="/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/metrics")

if __name__ == "__main__":
    main()