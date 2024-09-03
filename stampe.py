import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_results_detailed_copia.csv'
df = pd.read_csv(file_path)
file_path = 'validate_result_copia.csv'
dv=pd.read_csv(file_path)

combined_df  = pd.merge(df, dv, on=['Indice','Epoch', 'Learning Rate'])
combined_df ['Train Accuracy'] *= 100

indici = combined_df['Indice'].drop_duplicates()

plt.figure(figsize=(13, 6*len(indici)))


for i, indice  in enumerate(indici, start=1):
    # Filtraggio dei dati per la combinazione corrente
    filtered_df = combined_df.loc[combined_df['Indice'] == indice]

    # Estrarre il learning rate corrispondente
    lr = filtered_df['Learning Rate'].iloc[0]

    # Grafico della Loss per l'indice corrente
    plt.subplot(len(indici), 2, i*2-1)
    plt.plot(filtered_df['Epoch'], filtered_df['Train Loss'], marker='o', label='Train Loss', color='blue')
    plt.plot(filtered_df['Epoch'], filtered_df['Validation Loss'], marker='o', label='Validation Loss', color='orange')
    plt.title(f'Loss (Prova {indice} LR {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    # Grafico dell'Accuracy per l'indice corrente
    plt.subplot(len(indici), 2, i*2)
    plt.plot(filtered_df['Epoch'], filtered_df['Train Accuracy'], marker='o', label='Train Accuracy', color='blue')
    plt.plot(filtered_df['Epoch'], filtered_df['Validation Accuracy'], marker='o', label='Validation Accuracy', color='orange')
    plt.title(f'Accuracy (Prova {indice} LR {lr})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

plt.tight_layout()
plt.show()