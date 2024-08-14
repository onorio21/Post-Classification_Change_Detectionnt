import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    set_seed,
)

#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  #1024 originally
IMAGE_WIDTH = 240  #1024 originally
PIN_MEMORY = True
LOAD_MODEL = False 
TRAIN_IMG_DIR = ""
TRAIN_LABEL_DIR = ""
VAL_IMG_DIR = ""
VAL_LABEL_DIR = ""
SEED=42

set_seed(SEED)

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

TRAIN_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/train.txt"
VAL_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/val.txt"

def main():
    # Trasformazioni per il training senza augmentazione
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # Trasformazioni per la validazione (senza augmentazione)
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
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

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }  #save model
        save_checkpoint(checkpoint)
        
        check_accuracy(val_loader, model, device=DEVICE)  #check accuracy
        
        save_predictions_as_imgs(     #print some examples
            val_loader, model, folder="/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/provaraster", device=DEVICE
        )
    
if __name__ == "__main__":
    main()