import os
import pandas as pd
from gudhi.wasserstein import wasserstein_distance
import numpy as np
import csv
from itertools import product

# 📂 Directorio base
base_dir = "../5m/"
resultados_dirs = [f"resultados5m_{i}" for i in range(10, 110, 10)]
prefijos = sorted(next(os.walk(os.path.join(base_dir, resultados_dirs[0])))[1])

# 🛑 Lista de prefijos que quieres ignorar (cambia estos valores)
prefijos_ignorados = ["MCEFL"]

def leer_persistencia_csv(path):
    """Lee un archivo CSV y devuelve una lista de tuplas (dim, (birth, death))"""
    df = pd.read_csv(path)
    df["birth"] = df["birth"].round(5)
    df["death"] = df["death"].round(5)
    return [(int(row["dimension"]), (row["birth"], row["death"])) for _, row in df.iterrows()]

for prefijo in prefijos:
    if prefijo in prefijos_ignorados:
        print(f"⏩ Saltando prefijo: {prefijo}")
        continue

    print(f"🔄 Procesando prefijo: {prefijo}")
    output_file = f"distancias_wasserstein_{prefijo}.csv"

    with open(output_file, mode="w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "prefijo", "carpeta_a", "archivo_a", "carpeta_b", "archivo_b", "dimension", "distancia"
        ])
        writer.writeheader()

        # Mapear archivos por carpeta para este prefijo
        archivos_por_carpeta = {}
        for carpeta in resultados_dirs:
            ruta = os.path.join(base_dir, carpeta, prefijo)
            if not os.path.exists(ruta):
                continue
            archivos = sorted(f for f in os.listdir(ruta) if f.endswith(".csv"))
            archivos_por_carpeta[carpeta] = archivos

        # Comparar archivo a archivo directamente sin cargar todo
        for carpeta_a, carpeta_b in product(archivos_por_carpeta, repeat=2):
            for idx_a, archivo_a in enumerate(archivos_por_carpeta[carpeta_a]):
                ruta_a = os.path.join(base_dir, carpeta_a, prefijo, archivo_a)
                diag_a = leer_persistencia_csv(ruta_a)

                for idx_b, archivo_b in enumerate(archivos_por_carpeta[carpeta_b]):
                    ruta_b = os.path.join(base_dir, carpeta_b, prefijo, archivo_b)
                    diag_b = leer_persistencia_csv(ruta_b)

                    for dim in [0, 1, 2]:
                        diag_a_dim = np.array([
                            (birth, death) for d, (birth, death) in diag_a if d == dim
                        ])
                        diag_b_dim = np.array([
                            (birth, death) for d, (birth, death) in diag_b if d == dim
                        ])

                        if len(diag_a_dim) > 0 and len(diag_b_dim) > 0:
                            dist = wasserstein_distance(diag_a_dim, diag_b_dim, order=2., internal_p=2.)
                            writer.writerow({
                                "prefijo": prefijo,
                                "carpeta_a": carpeta_a,
                                "archivo_a": idx_a,
                                "carpeta_b": carpeta_b,
                                "archivo_b": idx_b,
                                "dimension": dim,
                                "distancia": dist
                            })

    print(f"✅ Prefijo {prefijo} procesado y guardado en {output_file}")

