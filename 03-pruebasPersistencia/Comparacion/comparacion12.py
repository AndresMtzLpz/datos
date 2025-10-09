import os
import pandas as pd
import numpy as np
import csv
from gudhi.wasserstein import wasserstein_distance

# -------------------------
# Parámetros globales
# -------------------------
MAX_PUNTOS = 72000  # máximo de puntos por dimensión
BASE_DIR = "../10m7/"  # directorio donde están las carpetas resultados10m6_X
PORCENTAJES = [f"resultados10m7_{i}" for i in range(10, 101, 10)]
OUTPUT_FILE = "distancias_wasserstein_10m7.csv"

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
        if len(puntos) > MAX_PUNTOS:
            print(f"⚠️  {os.path.basename(csv_path)} dim={dim}: se recortan {len(puntos)} → {MAX_PUNTOS}")
            puntos = puntos[:MAX_PUNTOS]
        diagramas[dim] = puntos
    return diagramas

# -------------------------
# Función para calcular y escribir distancias por dimensión
# -------------------------
def calcular_y_guardar(diag_a, diag_b, prefijo, carpeta_a, archivo_a, carpeta_b, archivo_b, writer):
    dims = sorted(set(diag_a.keys()) | set(diag_b.keys()))
    for dim in dims:
        if dim in diag_a and dim in diag_b and len(diag_a[dim]) > 0 and len(diag_b[dim]) > 0:
            dist = wasserstein_distance(diag_a[dim], diag_b[dim], order=2., internal_p=2.)
            writer.writerow({
                "prefijo": prefijo,
                "porcentaje_a": carpeta_a.split("_")[-1],
                "archivo_a": archivo_a,
                "porcentaje_b": carpeta_b.split("_")[-1],
                "archivo_b": archivo_b,
                "dimension": dim,
                "distancia": np.float32(dist)
            })
            print(f"✅ Distancia escrita: {prefijo} | {carpeta_a}-{archivo_a} vs {carpeta_b}-{archivo_b} | dim={dim} | dist={dist:.6f}")

# -------------------------
# Ejecución principal
# -------------------------
with open(OUTPUT_FILE, mode="w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=[
        "prefijo", "porcentaje_a", "archivo_a",
        "porcentaje_b", "archivo_b", "dimension", "distancia"
    ])
    writer.writeheader()

    print("🔄 Iniciando proceso global de cálculo Wasserstein...\n")

    for i, carpeta_a in enumerate(PORCENTAJES):
        ruta_a = os.path.join(BASE_DIR, carpeta_a)
        if not os.path.isdir(ruta_a):
            print(f"⚠️  Carpeta no encontrada: {carpeta_a}, se omite.")
            continue

        print(f"\n📂 Procesando carpeta A: {carpeta_a}")

        for prefijo in os.listdir(ruta_a):
            ruta_prefijo_a = os.path.join(ruta_a, prefijo)
            if not os.path.isdir(ruta_prefijo_a):
                continue

            print(f"   🔎 Prefijo: {prefijo}")

            archivos_a = [f for f in os.listdir(ruta_prefijo_a) if f.endswith(".csv")]
            for archivo_a in archivos_a:
                print(f"      📑 Archivo A: {archivo_a}")
                ruta_archivo_a = os.path.join(ruta_prefijo_a, archivo_a)
                diag_a = cargar_diagrama(ruta_archivo_a)

                for carpeta_b in PORCENTAJES[i:]:  # incluye comparaciones consigo mismo
                    ruta_prefijo_b = os.path.join(BASE_DIR, carpeta_b, prefijo)
                    if not os.path.isdir(ruta_prefijo_b):
                        continue

                    print(f"         🔗 Comparando con carpeta B: {carpeta_b}")
                    archivos_b = [f for f in os.listdir(ruta_prefijo_b) if f.endswith(".csv")]

                    for archivo_b in archivos_b:
                        print(f"            📑 Archivo B: {archivo_b}")
                        ruta_archivo_b = os.path.join(ruta_prefijo_b, archivo_b)
                        diag_b = cargar_diagrama(ruta_archivo_b)

                        calcular_y_guardar(
                            diag_a, diag_b,
                            prefijo, carpeta_a, archivo_a, carpeta_b, archivo_b,
                            writer
                        )

print(f"\n✅ Proceso terminado. Resultados guardados en {OUTPUT_FILE}")

