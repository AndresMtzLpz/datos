import pandas as pd
import numpy as np
from gudhi.wasserstein import wasserstein_distance
import csv
import os

# -------------------------
# Función para cargar un CSV como diagrama de persistencia
# -------------------------
def cargar_diagrama(csv_path):
    df = pd.read_csv(csv_path, dtype={
        "dimension": np.int32,
        "birth": np.float32,
        "death": np.float32
    })
    diagramas = {}
    for dim in df["dimension"].unique():
        puntos = df[df["dimension"] == dim][["birth", "death"]].to_numpy(dtype=np.float32)
        diagramas[dim] = puntos
    return diagramas

# -------------------------
# Función para calcular distancias y escribir resultados
# -------------------------
def calcular_y_guardar(diag_a, diag_b, archivo_a, archivo_b, writer=None, max_puntos=72000):
    dims = sorted(set(diag_a.keys()) | set(diag_b.keys()))
    for dim in dims:
        if dim in diag_a and dim in diag_b and len(diag_a[dim]) > 0 and len(diag_b[dim]) > 0:

            # --- limitar número máximo de puntos ---
            puntos_a = diag_a[dim][:max_puntos]
            puntos_b = diag_b[dim][:max_puntos]
            print(len(puntos_a))
            print(len(puntos_b))

            dist = wasserstein_distance(puntos_a, puntos_b, order=2., internal_p=2.)
            fila = {
                "archivo_a": os.path.basename(archivo_a),
                "archivo_b": os.path.basename(archivo_b),
                "dimension": dim,
                "distancia": np.float32(dist)
            }

            print(f"✅ {os.path.basename(archivo_a)} vs {os.path.basename(archivo_b)} | "
                  f"dim={dim} | usados={len(puntos_a)}x{len(puntos_b)} | distancia={dist:.6f}")

            if writer:
                writer.writerow(fila)

# -------------------------
# Archivos a comparar (¡tú los defines aquí!)
# -------------------------
archivos_a_comparar = [
    ("../10m6/resultados10m6_80/MIA/persistencia_MIA_componente_1.csv",
     "../10m6/resultados10m6_80/MIA/persistencia_MIA_componente_1.csv"),
]

# -------------------------
# Salida
# -------------------------
output_file = "distancias_wasserstein_manual.csv"

with open(output_file, mode="w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=["archivo_a", "archivo_b", "dimension", "distancia"])
    writer.writeheader()

    print("🔄 Iniciando cálculo manual de distancias Wasserstein...\n")

    for archivo_a, archivo_b in archivos_a_comparar:
        print(f"🔗 Comparando:\n   A: {archivo_a}\n   B: {archivo_b}")
        diag_a = cargar_diagrama(archivo_a)
        diag_b = cargar_diagrama(archivo_b)
        calcular_y_guardar(diag_a, diag_b, archivo_a, archivo_b, writer, max_puntos=72000)

print(f"\n✅ Proceso terminado. Resultados en: {output_file}")

