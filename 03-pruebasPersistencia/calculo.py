import os
import numpy as np
import matplotlib.pyplot as plt
from gudhi.cover_complex import MapperComplex
import gudhi as gd
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances

verbose = False

def detectar_prefijo(nombre_archivo, lista_prefijos):
    for prefijo in lista_prefijos:
        if nombre_archivo.startswith(prefijo):
            return prefijo.rstrip("_")
    return "Otro"

def cargar_y_unir_archivos_por_prefijos(directorio, lista_prefijos):
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
        df = pd.read_csv(path)
        df["archivo_origen"] = archivo
        df["prefijo_origen"] = detectar_prefijo(archivo, lista_prefijos)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

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

    # Prefijos a procesar (excluye prefijo2 y prefijo6)
    prefijos_a_usar = [p for i, p in enumerate(prefijos_todos) if i not in (2, 6)]

    output_directory = "datosEstandarizados3m_20/"  # <-- Carpeta de entrada

    for prefijo in prefijos_a_usar:
        print(f"\n🔹 Procesando prefijo: {prefijo}...")

        # Crear carpeta de salida para este prefijo
        carpeta_prefijo = os.path.join("resultados", prefijo)
        os.makedirs(carpeta_prefijo, exist_ok=True)

        # Cargar archivos de este prefijo
        df_union = cargar_y_unir_archivos_por_prefijos(output_directory, [prefijo])
        if df_union.empty:
            print(f"⚠️ No hay datos para prefijo {prefijo}, se salta.")
            continue

        # Dejar solo los valores numéricos
        df_numerico = df_union.select_dtypes(include='number').dropna()

        # Obtener matriz de distancia
        D = pairwise_distances(df_numerico.values)

        # Establecer cover
        cover_complex = MapperComplex(
            input_type='distance matrix', min_points_per_node=0,
            clustering=None, N=100, beta=0., C=10,
            filter_bnds=None, resolutions=[20, 2], gains=None, verbose=verbose
        )

        print("   → Construyendo complejidad Mapper...")
        _ = cover_complex.fit(D)

        # Guardar grafo PNG
        G = cover_complex.get_networkx()
        plt.figure()
        nx.draw(
            G,
            pos=nx.kamada_kawai_layout(G),
            node_color=[cover_complex.node_info_[v]["colors"][0] for v in G.nodes()]
        )
        plt.savefig(os.path.join(carpeta_prefijo, f"grafo_{prefijo}.png"))
        plt.close()

        # Guardar HTML
        cover_complex.data = df_numerico.values
        cover_complex.save_to_html(
            file_name=os.path.join(carpeta_prefijo, f"mapper_{prefijo}"),
            data_name=f"{prefijo}",
            cover_name="uniform",
            color_name="height"
        )

        # Calcular persistencia para componentes conexas > 2 nodos
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
                    rips_complex = gd.RipsComplex(points=component_points, sparse=0.3)
                    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
                    persistence = simplex_tree.persistence()

                    # Guardar PNG de persistencia
                    fig, ax = plt.subplots()
                    gd.plot_persistence_diagram(persistence, axes=ax)
                    ax.set_title(f"Persistencia - {prefijo} - Componente {i+1} ({len(comp)} nodos)")
                    plt.savefig(os.path.join(carpeta_prefijo, f"persistencia_{prefijo}_componente_{i+1}.png"))
                    plt.close(fig)

                    # Guardar CSV
                    df_persistence = pd.DataFrame(
                        [(dim, birth, death) for dim, (birth, death) in persistence],
                        columns=["dimension", "birth", "death"]
                    )
                    df_persistence.to_csv(
                        os.path.join(carpeta_prefijo, f"persistencia_{prefijo}_componente_{i+1}.csv"),
                        index=False
                    )

                    print(f"   → {prefijo} comp {i+1}: {len(comp)} nodos → {len(component_points)} puntos → Guardado.")

