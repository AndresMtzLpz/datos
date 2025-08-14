import os
import pandas as pd
import gudhi as gd
from itertools import product
from gudhi.wasserstein import wasserstein_distance
import numpy as np

# 📂 Directorio base
base_dir = "../"

# 📂 Carpetas de resultados (ordenadas)
resultados_dirs = [f"resultados3m_{i}" for i in range(10, 110, 10)]

# 📂 Prefijos detectados automáticamente
prefijos = sorted(next(os.walk(os.path.join(base_dir, resultados_dirs[0])))[1])

def leer_persistencia_csv(path):
    """Lee CSV y devuelve lista [(dim, (birth, death)), ...]"""
    df = pd.read_csv(path)
    return [(int(row["dimension"]), (row["birth"], row["death"])) for _, row in df.iterrows()]

# 📋 Lista para almacenar resultados
resultados_distancias = []

for prefijo in prefijos:
    # Guardar persistencias por carpeta
    persistencias_por_dir = {}
    for carpeta in resultados_dirs:
        ruta_prefijo = os.path.join(base_dir, carpeta, prefijo)
        if not os.path.exists(ruta_prefijo):
            continue
        archivos_csv = sorted(f for f in os.listdir(ruta_prefijo) if f.endswith(".csv"))
        persistencias_por_dir[carpeta] = [leer_persistencia_csv(os.path.join(ruta_prefijo, f)) for f in archivos_csv]

    # 🔄 Comparar todas las combinaciones (incluyendo misma carpeta)
    for carpeta_a, carpeta_b in product(persistencias_por_dir.keys(), repeat=2):
        lista_a = persistencias_por_dir[carpeta_a]
        lista_b = persistencias_por_dir[carpeta_b]

        for idx_a, diag_a in enumerate(lista_a):
            for idx_b, diag_b in enumerate(lista_b):
                for dim in [0, 1, 2]:
                    diag_a_dim = np.array([(birth, death) for d, (birth, death) in diag_a if d == dim])
                    diag_b_dim = np.array([(birth, death) for d, (birth, death) in diag_b if d == dim])

                    if len(diag_a_dim) > 0 and len(diag_b_dim) > 0:
                        dist_wass = wasserstein_distance(
                            diag_a_dim, diag_b_dim, order=2., internal_p=2.
                        )
                        resultados_distancias.append({
                            "prefijo": prefijo,
                            "carpeta_a": carpeta_a,
                            "archivo_a": idx_a,
                            "carpeta_b": carpeta_b,
                            "archivo_b": idx_b,
                            "dimension": dim,
                            "distancia": dist_wass
                        })

# 💾 Guardar resultados
df_resultados = pd.DataFrame(resultados_distancias)
df_resultados.to_csv("distancias_wasserstein_todas.csv", index=False)

print("✅ Distancias calculadas (incluyendo misma resolución) y guardadas en distancias_wasserstein_todas.csv")

