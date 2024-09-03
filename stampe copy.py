import pandas as pd
import matplotlib.pyplot as plt

file_path = 'training_results_detailed.csv'
df = pd.read_csv(file_path)
file_path = 'validate_result.csv'
dv=pd.read_csv(file_path)

# Fondere df e dv
combined_df = pd.merge(df, dv, on=['Indice', 'Epoch', 'Learning Rate'])

# Calcolare il valore massimo di F1 per ciascun indice
max_f1_values = combined_df.groupby('Indice')['F1'].max()

# Estrarre gli indici unici
indici = max_f1_values.index


# Estrarre i valori di F1 massimi
f1_values = max_f1_values.values


fig, axs = plt.subplots(1, figsize=(10, 5))

axs.bar(indici, f1_values)

for i, val in enumerate(f1_values):
    axs.text(i+1, val + 0.01, f'{val:.4f}', ha='center', va='bottom')



# Titoli e etichette
plt.title('Max F1 Values')
plt.xlabel('Indice')
plt.ylabel('F1')
plt.xticks(indici)
plt.ylim(0, max(f1_values) + 0.1)  # Impostare i limiti dell'asse y

plt.grid(False)
plt.tight_layout()
plt.show()