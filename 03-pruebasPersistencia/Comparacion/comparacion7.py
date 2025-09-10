import os
import pandas as pd
import gudhi as gd
from itertools import combinations_with_replacement, product
from gudhi.wasserstein import wasserstein_distance
import csv
import numpy as np

base_dir = "../5m2/"
resultados_dirs = [f"resultados5m_{i}" for i in range(10, 110, 10)]
prefijos = sorted(next(os.walk(os.path.join(base_dir, resultados_dirs[0])))[1])

def leer_persistencia_csv(path):
    """Lee un archivo CSV y devuelve una lista de tuplas (dim, (birth, death))"""
    df = pd.read_csv(path)
    df["birth"] = df["birth"].round(5)
    df["death"] = df["death"].round(5)
    return [(int(row["dimension"]), (row["birth"], row["death"])) for _, row in df.iterrows()]

for prefijo in prefijos:
    print(f"🔄 Procesando prefijo: {prefijo}")
    output_file = f"distancias_wasserstein_{prefijo}.csv"
    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "prefijo", "carpeta_a", "archivo_a", "carpeta_b", "archivo_b", "dimension", "distancia"
        ])
        writer.writeheader()
        
        # --- cargar archivos por carpeta ---
        archivos_por_carpeta = {
            carpeta: sorted(f for f in os.listdir(os.path.join(base_dir, carpeta, prefijo)) if f.endswith(".csv"))
            for carpeta in resultados_dirs
            if os.path.exists(os.path.join(base_dir, carpeta, prefijo))
        }
        
        # --- recorrer combinaciones de carpetas ---
        for carpeta_a, carpeta_b in combinations_with_replacement(archivos_por_carpeta, 2):
            archivos_a = archivos_por_carpeta.get(carpeta_a, [])
            archivos_b = archivos_por_carpeta.get(carpeta_b, [])
            if not archivos_a or not archivos_b:
                continue
            
            # precargar todos los diagramas en memoria
            diag_a_dict = {
                archivo: leer_persistencia_csv(os.path.join(base_dir, carpeta_a, prefijo, archivo))
                for archivo in archivos_a
            }
            diag_b_dict = {
                archivo: leer_persistencia_csv(os.path.join(base_dir, carpeta_b, prefijo, archivo))
                for archivo in archivos_b
            }
            
            # recorrer productos (archivo_a, archivo_b, dimension)
            for archivo_a, archivo_b, dim in product(archivos_a, archivos_b, [0, 1, 2]):
                diag_a_dim = np.array([(birth, death) for d, (birth, death) in diag_a_dict[archivo_a] if d == dim])
                diag_b_dim = np.array([(birth, death) for d, (birth, death) in diag_b_dict[archivo_b] if d == dim])
                
                if len(diag_a_dim) > 0 and len(diag_b_dim) > 0:
                    dist = wasserstein_distance(diag_a_dim, diag_b_dim, order=2., internal_p=2.)
                    writer.writerow({
                        "prefijo": prefijo,
                        "carpeta_a": carpeta_a,
                        "archivo_a": archivo_a,
                        "carpeta_b": carpeta_b,
                        "archivo_b": archivo_b,
                        "dimension": dim,
                        "distancia": dist
                    })
    
    print(f"✅ Prefijo {prefijo} procesado y guardado en {output_file}")

