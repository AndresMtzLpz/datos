import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gudhi.cover_complex import MapperComplex
import gudhi as gd
import networkx as nx

from kmapper import KeplerMapper, Cover
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# ---------------------------
# Funciones auxiliares
# ---------------------------

def detectar_prefijo(nombre_archivo, lista_prefijos):
    for prefijo in lista_prefijos:
        if nombre_archivo.startswith(prefijo):
            return prefijo.rstrip("_")
    return "Otro"

def contar_lineas_3m(directorio_3m, lista_prefijos):
    """Cuenta el número de líneas en los archivos .log de 3m por prefijo"""
    lineas_por_prefijo = {}
    if not os.path.exists(directorio_3m):
        print(f"⚠️ Carpeta {directorio_3m} no encontrada.")
        return lineas_por_prefijo

    for prefijo in lista_prefijos:
        archivos = [
            f for f in os.listdir(directorio_3m)
            if f.startswith(prefijo) and f.endswith(".log")
        ]
        total_lineas = 0
        for archivo in archivos:
            with open(os.path.join(directorio_3m, archivo), "r") as f:
                total_lineas += sum(1 for _ in f) - 1  # quitar encabezado
        lineas_por_prefijo[prefijo] = total_lineas
    return lineas_por_prefijo

def cargar_y_unir_archivos_por_prefijos_limitado_random(directorio, lista_prefijos, limites, seed):
    """Carga los archivos de 5m pero seleccionando aleatoriamente N líneas (según límite de 3m)"""
    if not os.path.exists(directorio):
        print(f"⚠️ Carpeta {directorio} no encontrada.")
        return pd.DataFrame()

    archivos = os.listdir(directorio)
    archivos_filtrados = [
        f for f in archivos
        if any(f.startswith(prefijo) for prefijo in lista_prefijos)
        and f.endswith('.log')
    ]
    if not archivos_filtrados:
        print("⚠️ No se encontraron archivos con los prefijos indicados.")
        return pd.DataFrame()

    dataframes = []
    for archivo in archivos_filtrados:
        path = os.path.join(directorio, archivo)
        prefijo = detectar_prefijo(archivo, lista_prefijos)
        n_lineas = limites.get(prefijo, None)

        # cargar todo el archivo
        df_completo = pd.read_csv(path)
        if n_lineas and len(df_completo) > n_lineas:
            df = df_completo.sample(n=n_lineas, random_state=seed)  # semilla diferente por iteración
        else:
            df = df_completo

        df["archivo_origen"] = archivo
        df["prefijo_origen"] = prefijo
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# ---------------------------
# Script principal
# ---------------------------
if __name__ == "__main__":
    # Define prefijos
    prefijos_todos = [
        'MCEFL',  # 0
        'MIA',    # 1
        'MPFC',   # 2 (excluir)
        'MRS',    # 3
        'MSC',    # 4
        'MSFC',   # 5
        'MVIE',   # 6 (excluir)
        'MVIV',   # 7
        'WIDS',   # 8
        'WIDSL',  # 9
        'WLEC',   # 10
        'WSUT',   # 11
        'WVAE',   # 12
        'WVAL',   # 13
        'WVIV'    # 14
    ]

    # Excluir prefijos 2 y 6
    prefijos_a_usar = [p for i, p in enumerate(prefijos_todos) if i not in (2, 6)]

    # Rutas a carpetas con logs (ajusta las rutas a tu entorno)
    ruta_3m = "3m/datosEstandarizados3m_100"
    ruta_5m = "4m/datosEstandarizados4m_100"

    # Carpeta de salida para los HTML
    carpeta_salida = "resultados_grafos"
    os.makedirs(carpeta_salida, exist_ok=True)

    # Paso 1: Obtener límites por prefijo desde 3m
    limites_por_prefijo = contar_lineas_3m(ruta_3m, prefijos_a_usar)

    # Paso 2: Cargar datos de 5m con los mismos límites por prefijo
    df_5m = cargar_y_unir_archivos_por_prefijos_limitado_random(
        directorio=ruta_5m,
        lista_prefijos=prefijos_a_usar,
        limites=limites_por_prefijo,
        seed=42
    )

    if df_5m.empty:
        print("⚠️ No se pudo construir el grafo. El DataFrame está vacío.")
    else:
        # Paso 3: Preparar datos numéricos
        X = df_5m.select_dtypes(include=[np.number]).dropna(axis=1)
        X = StandardScaler().fit_transform(X)

        # Paso 4: Crear Mapper y aplicar proyección (PCA)
        mapper = KeplerMapper(verbose=0)
        lens = mapper.fit_transform(X, projection=PCA(n_components=13))

        # Parámetros a probar
        n_cubes_list = [30,40,50]
        overlap_list = [0.1, 0.2]
        eps_list = [2, 3,4]
        min_samples_list = [13]

        combinaciones_generadas = 0

        for n_cubes in n_cubes_list:
            for overlap in overlap_list:
                for eps in eps_list:
                    for min_samples in min_samples_list:
                        nombre_archivo = f"grafo_mapper_cubes{n_cubes}_overlap{int(overlap*10):02}_eps{eps}_min{min_samples}.html"
                        ruta_archivo = os.path.join(carpeta_salida, nombre_archivo)
                        print(f"🔧 Generando: {ruta_archivo}")

                        cover = Cover(n_cubes=n_cubes, perc_overlap=overlap)
                        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

                        try:
                            graph = mapper.map(
                                lens,
                                X,
                                cover=cover,
                                clusterer=clusterer
                            )

                            mapper.visualize(
                                graph,
                                path_html=ruta_archivo,
                                title=f"Grafo Topológico (cubes={n_cubes}, overlap={overlap}, eps={eps}, min={min_samples})",
                                color_values=df_5m["prefijo_origen"].astype("category").cat.codes,
                                color_function_name="Prefijo",
                                custom_tooltips=df_5m["prefijo_origen"]
                            )

                            print(f"✅ Grafo guardado en: {ruta_archivo}")
                            combinaciones_generadas += 1

                            # Limitar a 12 combinaciones si solo quieres ~10
                            if combinaciones_generadas >= 12:
                                raise StopIteration

                        except Exception as e:
                            print(f"❌ Error en combinación: {e}")

        print(f"\n✅ Se generaron {combinaciones_generadas} grafos topológicos en la carpeta '{carpeta_salida}'.")

