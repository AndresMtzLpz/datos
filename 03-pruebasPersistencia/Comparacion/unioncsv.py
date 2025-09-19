import os
import pandas as pd

# Carpeta donde están los CSV
carpeta = "./"
archivos = [f for f in os.listdir(carpeta) if f.startswith("distancias_wasserstein_10m1") and f.endswith(".csv")]

# Lista para guardar los DataFrames
dataframes = []

for archivo in archivos:
    ruta = os.path.join(carpeta, archivo)
    df = pd.read_csv(ruta)
    dataframes.append(df)

# Concatenar todos los DataFrames
df_final = pd.concat(dataframes, ignore_index=True)

# Guardar en un solo CSV
df_final.to_csv("distancias_wasserstein_10m1.csv", index=False)

print(f"Se han unido {len(archivos)} archivos en 'distancias_wasserstein_10m.csv'")

