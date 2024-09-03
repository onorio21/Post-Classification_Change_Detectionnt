import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_results_detailed.csv'
df = pd.read_csv(file_path)
file_path = 'validate_result.csv'
dv=pd.read_csv(file_path)

combined_df  = pd.merge(df, dv, on=['Epoch', 'Learning Rate'])
combined_df ['Train Accuracy'] *= 100

epoche = combined_df['Epoch'].drop_duplicates()

plt.figure(figsize=(10, 12))


# Grafico della Loss per l'indice corrente

plt.subplot(2,1,1)
plt.plot(combined_df['Epoch'], combined_df['Train Loss'], marker='o', label='Train Loss', color='blue')
plt.plot(combined_df['Epoch'], combined_df['Validation Loss'], marker='o', label='Validation Loss', color='orange')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

 # Grafico dell'Accuracy per l'indice corrente
plt.subplot(2,1,2)
plt.plot(combined_df['Epoch'], combined_df['Train Accuracy'], marker='o', label='Train Accuracy', color='blue')
plt.plot(combined_df['Epoch'], combined_df['Validation Accuracy'], marker='o', label='Validation Accuracy', color='orange')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)

plt.tight_layout()
plt.show()