import os
import pandas as pd
import gudhi as gd
from itertools import product
from gudhi.wasserstein import wasserstein_distance
import numpy as np
import csv

# 📂 Directorio base
base_dir = "../5m/"

# 📂 Carpetas de resultados (ordenadas)
resultados_dirs = [f"resultados5m_{i}" for i in range(10, 110, 10)]

# 📂 Prefijos detectados automáticamente
prefijos = sorted(next(os.walk(os.path.join(base_dir, resultados_dirs[0])))[1])

def leer_persistencia_csv(path):
    """Lee un archivo CSV y devuelve una lista de tuplas (dim, (birth, death))"""
    df = pd.read_csv(path)
    return [(int(row["dimension"]), (row["birth"], row["death"])) for _, row in df.iterrows()]

# 🔁 Procesar cada prefijo individualmente
for prefijo in prefijos:
    print(f"🔄 Procesando prefijo: {prefijo}")

    # Crear archivo de salida por prefijo
    output_file = f"distancias_wasserstein_{prefijo}.csv"
    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "prefijo", "carpeta_a", "archivo_a", "carpeta_b", "archivo_b", "dimension", "distancia"
        ])
        writer.writeheader()

        # Cargar persistencias por carpeta
        persistencias_por_dir = {}
        for carpeta in resultados_dirs:
            ruta_prefijo = os.path.join(base_dir, carpeta, prefijo)
            if not os.path.exists(ruta_prefijo):
                continue
            archivos_csv = sorted(f for f in os.listdir(ruta_prefijo) if f.endswith(".csv"))
            persistencias_por_dir[carpeta] = [
                leer_persistencia_csv(os.path.join(ruta_prefijo, f)) for f in archivos_csv
            ]

        # Comparar todas las combinaciones
        for carpeta_a, carpeta_b in product(persistencias_por_dir.keys(), repeat=2):
            lista_a = persistencias_por_dir[carpeta_a]
            lista_b = persistencias_por_dir[carpeta_b]

            for idx_a, diag_a in enumerate(lista_a):
                for idx_b, diag_b in enumerate(lista_b):
                    for dim in [0, 1, 2]:
                        diag_a_dim = np.array([
                            (birth, death) for d, (birth, death) in diag_a if d == dim
                        ])
                        diag_b_dim = np.array([
                            (birth, death) for d, (birth, death) in diag_b if d == dim
                        ])

                        if len(diag_a_dim) > 0 and len(diag_b_dim) > 0:
                            dist_wass = wasserstein_distance(
                                diag_a_dim, diag_b_dim, order=2., internal_p=2.
                            )
                            writer.writerow({
                                "prefijo": prefijo,
                                "carpeta_a": carpeta_a,
                                "archivo_a": idx_a,
                                "carpeta_b": carpeta_b,
                                "archivo_b": idx_b,
                                "dimension": dim,
                                "distancia": dist_wass
                            })

        del persistencias_por_dir  # Liberar memoria

print("✅ Todos los prefijos procesados por separado.")

