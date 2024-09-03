import random
import torch
import torchvision
import os
import rasterio
from rasterio.transform import from_origin
import numpy as np
from dataset import SN7BuildingsDataset
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(state, filename="my_checkpoint.pht.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_txt,
        val_txt,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_ds = SN7BuildingsDataset(
        txt_file=train_txt,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = SN7BuildingsDataset(
        txt_file=val_txt,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='mps'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

#def save_predictions_as_imgs(loader, model, folder='/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/provaraster', device='cpu'):
#    # Ensure the folder exists, if not, create it
#    if not os.path.exists(folder):
#        os.makedirs(folder)
#    
#    model.eval()
#    for idx, (x, y) in enumerate(loader):
#        x = x.to(device=device)
#        with torch.no_grad():
#            preds = torch.sigmoid(model(x))
#            preds = (preds > 0.5).float()
#        
#        # Resize predictions to 1024x1024
#        preds = torchvision.transforms.functional.resize(preds, [1024, 1024])
#
#        # Convert the predictions tensor to numpy array
#        preds_np = preds.squeeze().cpu().numpy()
#
#        # Define the transform (identity transform here, you can adjust based on your geospatial data)
#        transform = from_origin(0, 0, 1, 1)  # This is a placeholder, adjust as needed for your data
#
#        # Save the prediction as a TIFF file
#        tiff_path = os.path.join(folder, f"pred_{idx}.tif")
#        with rasterio.open(
#            tiff_path,
#            'w',
#            driver='GTiff',
#            height=preds_np.shape[0],
#            width=preds_np.shape[1],
#            count=1,  # Number of bands
#            dtype=preds_np.dtype,
#            crs='+proj=latlong',  # You can specify your CRS here if needed
#            transform=transform
#        ) as dst:
#            dst.write(preds_np, 1)
#    model.train()
        
        

#def save_predictions_as_imgs(
#        loader, model, folder='/Users/onorio21/Desktop/preds', device='mps'
#):
#    model.eval()
#    for idx, (x, y) in enumerate(loader):
#        x = x.to(device=device)
#        y = y.to(device=device)  # Ensure y is also on the same device
#
#        with torch.no_grad():
#            preds = (model(x))
#            preds = (preds > 0.5).float()
#
#        # Iterate over the batch and save each prediction individually
#        for i in range(preds.shape[0]):
#            torchvision.utils.save_image(
#                preds[i].unsqueeze(0), f"{folder}/pred_{idx}_{i}.png"
#            )
#            torchvision.utils.save_image(
#                y[i].unsqueeze(0), f"{folder}/{idx}_{i}.png"
#            )
#
#    model.train()


def save_predictions_as_imgs(
        loader, model, folder='/Users/onorio21/Desktop/preds', device='mps'
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)  # Ensure y is also on the same device

        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()

        # Iterate over the batch and save each prediction individually
        for i in range(preds.shape[0]):
            # Invert the predicted and ground truth masks
            inverted_preds = 1 - preds[i]
            inverted_y = 1 - y[i]

            # Save the inverted predictions and ground truth masks
            torchvision.utils.save_image(
                inverted_preds.unsqueeze(0), f"{folder}/pred_{idx}_{i}.png"
            )
            torchvision.utils.save_image(
                inverted_y.unsqueeze(0), f"{folder}/{idx}_{i}.png"
            )

    model.train()
