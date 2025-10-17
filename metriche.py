import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Impostazioni
base_path = './._0414_1333'  
output_tsv = 'metriche_best_config.tsv'

metriche = [
    'valid_f1', 'valid_acc', 'valid_prec', 'valid_rec',
    'test_f1', 'test_acc', 'test_prec', 'test_rec'
]

# Raccolta dati
dati = []

for categoria in os.listdir(base_path):
    percorso_categoria = os.path.join(base_path, categoria)
    percorso_config = os.path.join(percorso_categoria, 'best_config.json')

    if os.path.isdir(percorso_categoria) and os.path.isfile(percorso_config):
        with open(percorso_config, 'r') as f:
            config = json.load(f)
        record = {'categoria': categoria}
        for metrica in metriche:
            record[metrica] = round(config.get(metrica, 0), 4) 
        dati.append(record)

# Crea DataFrame ordinato
df = pd.DataFrame(dati)
df = df.sort_values(by='categoria')

# Salva in TSV leggibile
df.to_csv(output_tsv, sep='\t', index=False)
print(f'Dati salvati in formato TSV: {output_tsv}')

# (opzionale) mostra anteprima
print("\nAnteprima del file TSV:\n")
print(df.to_string(index=False))
