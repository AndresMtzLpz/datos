import os
import numpy as np
import matplotlib.pyplot as plt
from gudhi.cover_complex import MapperComplex
import gudhi as gd
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances

verbose = False

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
            df = df_completo.sample(n=n_lineas, random_state=seed)
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
    prefijos_todos = [
        'MCEFL',  # prefijo0
        'MIA',    # prefijo1
        'MPFC',   # prefijo2 (excluir)
        'MRS',    # prefijo3
        'MSC',    # prefijo4
        'MSFC',   # prefijo5
        'MVIE',   # prefijo6 (excluir)
        'MVIV',   # prefijo7
        'WIDS',   # prefijo8
        'WIDSL',  # prefijo9
        'WLEC',   # prefijo10
        'WSUT',   # prefijo11
        'WVAE',   # prefijo12
        'WVAL',   # prefijo13
        'WVIV'    # prefijo14
    ]

    # Prefijos a procesar (excluye varios)
    prefijos_a_usar = [p for i, p in enumerate(prefijos_todos) if i not in (2,6)]
    print(prefijos_a_usar)

    # Repetir 4 veces
    for iteracion in range(4, 6):
        print(f"\n============================")
        print(f"  🔁 Iteración {iteracion}")
        print(f"============================")

        # Procesar para 10, 20, ..., 100
        for n in range(10, 110, 10):
            print(f"\n📂 Nivel {n}")

            carpeta_3m = f"3m/datosEstandarizados3m_{n}/"
            carpeta_5m = f"10m/datosEstandarizados10m_{n}/"
            carpeta_resultados = f"resultados10mt{iteracion}_{n}/"
            os.makedirs(carpeta_resultados, exist_ok=True)

            # calcular límites de líneas por prefijo con base en 3m
            limites_lineas = contar_lineas_3m(carpeta_3m, prefijos_a_usar)

            for prefijo in prefijos_a_usar:
                print(f"   🔹 Procesando prefijo: {prefijo}")

                # Crear carpeta de salida para este prefijo
                carpeta_prefijo = os.path.join(carpeta_resultados, prefijo)
                os.makedirs(carpeta_prefijo, exist_ok=True)

                # Cargar datos de 5m limitados al número de líneas de 3m (aleatoriamente, semilla depende de iteración)
                df_union = cargar_y_unir_archivos_por_prefijos_limitado_random(
                    carpeta_5m, [prefijo], limites_lineas, seed=iteracion
                )
                if df_union.empty:
                    print(f"⚠️ No hay datos para prefijo {prefijo}, se salta.")
                    continue

                # Datos numéricos
                df_numerico = df_union.select_dtypes(include='number').dropna()

                # Si el tamaño no es suficiente, saltar
                N_param = 600
                if len(df_numerico) < N_param:
                    print(f"⚠️ Prefijo {prefijo} tiene solo {len(df_numerico)} filas (< {N_param}), se salta.")
                    continue

                # Matriz de distancias
                print("      → Calculo matriz de distancia")
                D = pairwise_distances(df_numerico.values.astype(np.float32))

                # Cover complex
                cover_complex = MapperComplex(
                    input_type='distance matrix', min_points_per_node=0,
                    clustering=None, N=N_param, beta=0., C=60,
                    filter_bnds=None, resolutions=[20, 2], gains=None, verbose=verbose
                )

                print("      → Calculo cover complex")
                try:
                    _ = cover_complex.fit(D)
                except ValueError as e:
                    print(f"⚠️ Error en cover_complex.fit para {prefijo}: {e}. Se salta.")
                    continue

                # Grafo PNG
                G = cover_complex.get_networkx()
                plt.figure()
                nx.draw(
                    G,
                    pos=nx.kamada_kawai_layout(G),
                    node_color=[cover_complex.node_info_[v]["colors"][0] for v in G.nodes()]
                )
                plt.savefig(os.path.join(carpeta_prefijo, f"grafo_{prefijo}.png"))
                plt.close()

                # HTML
                cover_complex.data = df_numerico.values
                cover_complex.save_to_html(
                    file_name=os.path.join(carpeta_prefijo, f"mapper_{prefijo}"),
                    data_name=f"{prefijo}",
                    cover_name="uniform",
                    color_name="height"
                )

                # Persistencia por componente conexa (> 4 nodos)
                componentes = list(nx.connected_components(G))
                for i, comp in enumerate(componentes):
                    if len(comp) > 4:
                        all_point_indices = []
                        for node in comp:
                            point_indices = cover_complex.node_info_[node]["indices"]
                            all_point_indices.extend(point_indices)

                        all_point_indices = np.unique(all_point_indices)
                        component_points = df_numerico.values[all_point_indices]

                        if len(component_points) > 0:
                            rips_complex = gd.RipsComplex(points=component_points, sparse=0.6)
                            simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
                            persistence = simplex_tree.persistence()

                            # PNG persistencia
                            fig, ax = plt.subplots()
                            gd.plot_persistence_diagram(persistence, axes=ax)
                            ax.set_title(f"Persistencia - {prefijo} - Componente {i+1} ({len(comp)} nodos)")
                            plt.savefig(os.path.join(carpeta_prefijo, f"persistencia_{prefijo}_componente_{i+1}.png"))
                            plt.close(fig)

                            # CSV persistencia
                            df_persistence = pd.DataFrame(
                                [(dim, birth, death) for dim, (birth, death) in persistence],
                                columns=["dimension", "birth", "death"]
                            )
                            df_persistence.to_csv(
                                os.path.join(carpeta_prefijo, f"persistencia_{prefijo}_componente_{i+1}.csv"),
                                index=False
                            )

                            print(f"      → {prefijo} comp {i+1}: {len(comp)} nodos → {len(component_points)} puntos → Guardado.")

