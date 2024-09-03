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
from PIL import Image
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import csv

from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    set_seed,
)

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 15
IMAGE_HEIGHT = 256  # 1024 originally
IMAGE_WIDTH = 256  # 1024 originally
LOAD_MODEL = False 
SEED = 42
TRAIN_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/train.txt"
VAL_TXT = "/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits/val.txt"
# Assuming these values are precomputed or calculated beforehand
num_positives = 68454710  # Example value, replace with actual count
num_negatives = 964053738  # Example value, replace with actual count
differenza=num_negatives/num_positives

loss_fn = torch.nn.CrossEntropyLoss(weight = torch.tensor([0.07,1]).to(DEVICE))

def acc_metric(predb, yb):
  return (predb.argmax(dim=1) == yb.to(DEVICE)).float().mean()

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()


def train(indice, model, device, optimizer, loss_fn, lr, epochs, train_loader, val_loader, validate_filename, acc_fn, log_interval=5):
    # Set model to train mode
    print("sono nel train \n")
    model.train()
    train_losses = []
    train_acc = []
    best_f1 = -1000
    best_model_state_dict = None

    # Training loop
    for epoch in  range(epochs):
      total_loss = 0
      accuracy = 0
      for batch_idx, (x, target_label) in enumerate(train_loader):
           print("sono nel secondo for \n")
           x=x.to(device)
           target_label=target_label.to(device).long()

           optimizer.zero_grad()
           output = model(x)
           loss = loss_fn(output, target_label)
           loss.backward()
           optimizer.step()

           total_loss += loss.item()*train_loader.batch_size # in questo modo peso la loss in base al batch
           acc = (acc_fn(output, target_label).item())*train_loader.batch_size # in questo modo peso l'accuratezza in base al batch
           accuracy += acc
           print("fatto secondo for \n")
           if batch_idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(x),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()
              ))

      print("sono furoi dal primo batch")
      train_losses.append(total_loss / len(train_loader.dataset))
      train_acc.append(accuracy / len(train_loader.dataset))

      print(f'Accuracy-> {(accuracy / len(train_loader.dataset)) * 100:.2f}%')

      val_loss, val_acc, precision, recall, f1_score = validate(model, device, loss_fn, val_loader, acc_fn)

      if best_f1 < f1_score:
        best_f1 = f1_score
        print("cambio best f1->",best_f1 )
        best_model_state_dict = model.state_dict()


      with open(validate_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([indice, lr,  epoch + 1, val_loss, val_acc, precision, recall, f1_score])



    print("=> Salvataggio checkpoint")
    torch.save({'model_state_dict':best_model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': f1_score
                }, f'{indice}model.pt')

    print("salvataggio avvennuto")

    return train_losses, train_acc

def flatten(xss):
    return [x for xs in xss for x in xs]

def validate(model, device, loss_fn, validation_loader, acc_fn):
    print("inizio validazione")
    model.eval()
    val_loss, correct = 0, 0
    y_pred = []
    y_true = []

    # Disabilito il calcolo del gradiente per velocizzare
    with torch.no_grad():
        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)
            
            # Calcolo output usando i dati come input per la rete
            outputs = model(x)
            loss = loss_fn(outputs, y.long())
            
            # Calcolo loss usando criterion
            val_loss += loss.item() * validation_loader.batch_size
            
            # Calcolo accuratezza
            acc = (acc_fn(outputs, y).item()) * validation_loader.batch_size
            correct += acc

            #print(f"y shape: {y.shape}, output shape: {outputs.shape}")

            # Convert predictions to class indices and append them to the lists
            y_true.extend(y.cpu().numpy().flatten())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy().flatten())

    print("calcolo recall, f1 e precision")

    # Ensure both lists are the same length before calculating metrics
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # Calcola la media della loss e dell'accuratezza sul set di validazione
    val_loss /= len(validation_loader.dataset)
    accuracy = (correct / len(validation_loader.dataset)) * 100

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), F1-score: {:.4f}\n'.format(
    val_loss,
    correct,
    len(validation_loader.dataset),
    accuracy,
    f1
    ))
    return val_loss, accuracy, precision, recall, f1



def main():
    print("sono nel main")
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),  # Flip the image horizontally with a 50% probability
            A.VerticalFlip(p=0.5),  # Flip the image vertically with a 50% probability
            A.RandomRotate90(p=0.5),  # Randomly rotate the image by 90 degrees
            #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),  # Random shifts, scaling, and rotation
            #A.RandomBrightnessContrast(p=0.2),  # Randomly adjust brightness and contrast
            #A.HueSaturationValue(p=0.3),  # Adjust hue, saturation, and value
            #A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),  # Shift the RGB channels
            #A.ElasticTransform(p=0.2),  # Apply elastic deformation
            #A.Perspective(p=0.2),  # Apply perspective transformation
            #A.GridDistortion(p=0.2),  # Apply grid distortion
            #A.GaussianBlur(p=0.2),  # Apply Gaussian blur
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )

    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )
    train_loader, val_loader = get_loaders(
        TRAIN_TXT,
        VAL_TXT,
        BATCH_SIZE,
        train_transform,
        val_transform,
    )
    
    learning_rates1 = [1e-4]
    num_epochs=15

    #creo i due file csv dove salvo i dati
    print("creo i file")
    results_filename = 'training_results_detailed.csv'
    validate_filename = 'validate_result.csv'

    with open(results_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Indice','Learning Rate', 'Epoch', 'Train Loss', 'Train Accuracy'])

    with open(validate_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Indice', 'Learning Rate', 'Epoch', 'Validation Loss', 'Validation Accuracy', 'Precision', 'Recall', 'F1'])


    i=1

    for lr in learning_rates1:
        print("inizio training per il modello con lr:", lr)
        model= UNET(in_channels=3, out_channels=2)
        model=model.to(DEVICE)


        optimizer=optim.Adam(model.parameters(), lr=lr)

        train_losses, train_acc = train(i, model, DEVICE, optimizer, loss_fn, lr, num_epochs, train_loader, val_loader, validate_filename, acc_metric)

        save_predictions_as_imgs(
                val_loader, model, folder='/Users/onorio21/Desktop/preds', device=DEVICE
            )

        val_loader_iter = iter(val_loader)

        with torch.no_grad():
            for _ in range(5):  # Visualizza i risultati per 5 batch
                xb, yb = next(val_loader_iter)
                predb = model(xb.to(DEVICE))

                fig, ax = plt.subplots(4, 3, figsize=(15, 4*5))  # Modify to 3 columns
                for i in range(4):
                    ax[i, 0].imshow(batch_to_img(xb, i))
                    ax[i, 0].set_title("Input Image")
                    
                    ax[i, 1].imshow(predb_to_mask(predb, i), cmap='gray')
                    ax[i, 1].set_title("Predicted Mask")
                    
                    ax[i, 2].imshow(yb[i].cpu().numpy(), cmap='gray')  # Display the ground truth mask
                    ax[i, 2].set_title("Ground Truth Mask")
                    
                plt.tight_layout()
                plt.show()

        with open(results_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            for epoch in range(num_epochs):
                writer.writerow([i, lr,  epoch + 1, train_losses[epoch], train_acc[epoch]])

        #plot_and_save_metrics(train_losses, val_losses, val_metrics, output_dir="/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/metrics")

            i+=1

if __name__ == "__main__":
    main()