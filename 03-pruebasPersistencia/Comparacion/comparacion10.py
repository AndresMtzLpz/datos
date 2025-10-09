import os
import pandas as pd
import numpy as np
import csv
from gudhi.wasserstein import wasserstein_distance

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
            print(f"✅ Distancia escrita: {prefijo} | {carpeta_a}-{archivo_a} vs {carpeta_b}-{archivo_b} | dim={dim}")

# -------------------------
# Configuración
# -------------------------
base_dir = "../10m6/"  # directorio donde están las carpetas resultados10m6_X
porcentajes = [f"resultados10m6_{i}" for i in range(10, 101, 10)]

output_file = "distancias_wasserstein_10m6.csv"

with open(output_file, mode="w", newline="") as f_out:
    writer = csv.DictWriter(f_out, fieldnames=[
        "prefijo", "porcentaje_a", "archivo_a", "porcentaje_b", "archivo_b", "dimension", "distancia"
    ])
    writer.writeheader()

    print("🔄 Iniciando proceso...")

    for i, carpeta_a in enumerate(porcentajes):
        ruta_a = os.path.join(base_dir, carpeta_a)
        if not os.path.isdir(ruta_a):
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
                diag_a = cargar_diagrama(os.path.join(ruta_prefijo_a, archivo_a))
                
                for carpeta_b in porcentajes[i:]:  # incluye comparaciones consigo mismo
                    ruta_b = os.path.join(base_dir, carpeta_b, prefijo)
                    if not os.path.isdir(ruta_b):
                        continue
                    
                    print(f"         🔗 Comparando con carpeta B: {carpeta_b}")
                    archivos_b = [f for f in os.listdir(ruta_b) if f.endswith(".csv")]
                    
                    for archivo_b in archivos_b:
                        print(f"            📑 Archivo B: {archivo_b}")
                        diag_b = cargar_diagrama(os.path.join(ruta_b, archivo_b))
                        
                        # calcular y escribir por dimensión inmediatamente
                        calcular_y_guardar(
                            diag_a, diag_b,
                            prefijo, carpeta_a, archivo_a, carpeta_b, archivo_b,
                            writer
                        )

print("\n✅ Proceso terminado. Resultados en distancias_wasserstein.csv")

