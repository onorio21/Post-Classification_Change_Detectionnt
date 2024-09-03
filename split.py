import os
import numpy as np

def create_split_files(image_root_dir, label_root_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, N=None):
    # Ottieni tutte le sottocartelle in train (immagini)
    image_subfolders = [os.path.join(image_root_dir, folder) for folder in os.listdir(image_root_dir)]
    
    if N:
        image_subfolders = image_subfolders[:N]
    
    all_pairs = []
    
    for image_subfolder in image_subfolders:
        # Ottieni il nome della cartella corrispondente in conversions (labels)
        subfolder_name = os.path.basename(image_subfolder)
        label_subfolder = os.path.join(label_root_dir, subfolder_name)
        
        image_folder = os.path.join(image_subfolder, 'images_masked')
        label_folder = label_subfolder  # Labels sono direttamente nella sottocartella corrispondente

        print(f"Esaminando le cartelle: {image_folder} e {label_folder}")
        
        if os.path.exists(image_folder) and os.path.exists(label_folder):
            for tif_file in os.listdir(image_folder):
                if tif_file.endswith('.tif'):
                    img_path = os.path.join(image_folder, tif_file)
                    label_path = os.path.join(label_folder, tif_file)
                    if os.path.exists(label_path):
                        all_pairs.append(f"{img_path} {label_path}\n")
                        print(f"Abbinata immagine e label: {img_path}, {label_path}")
                    else:
                        print(f"Label non trovata per: {img_path}")
                else:
                    print(f"File non supportato trovato in {image_folder}: {tif_file}")
        else:
            print(f"Cartelle 'images_masked' o 'labels_match' mancanti in: {subfolder_name}")
    
    if not all_pairs:
        print("Nessuna coppia immagine-label trovata!")
        return
    
    np.random.shuffle(all_pairs)
    
    train_split = int(len(all_pairs) * train_ratio)
    val_split = int(len(all_pairs) * (train_ratio + val_ratio))
    
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.writelines(all_pairs[:train_split])
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.writelines(all_pairs[train_split:val_split])
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.writelines(all_pairs[val_split:])
    
    print(f"Creati file di split: train.txt ({train_split} campioni), val.txt ({val_split - train_split} campioni), test.txt ({len(all_pairs) - val_split} campioni)")

# Esempio di esecuzione
image_root_dir = '/Users/onorio21/Desktop/Università/train'
label_root_dir = '/Users/onorio21/Desktop/Università/conversions'
output_dir = '/Users/onorio21/Desktop/Università/Laboratorio AI/Post-Classification_Change_Detectionnt/splits'
create_split_files(image_root_dir, label_root_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, N=None)