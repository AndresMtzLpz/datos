import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import networkx as nx
import kmapper as km
import gudhi as gd
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # necesario para 3D
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

def plot_persistence_diagram_gudhi(persistence, idx, output_dir):
    plt.figure(figsize=(6, 6))

    # Scatter por dimensión
    for dim in set(dim for dim, _ in persistence):
        points = [(b, d) for d_dim, (b, d) in persistence if d_dim == dim and d != float('inf')]
        if points:
            points = np.array(points)
            plt.scatter(points[:, 0], points[:, 1], label=f"H{dim}", s=30)

    # Diagonal
    all_bd = [b for _, (b, _) in persistence] + [d for _, (_, d) in persistence if d != float('inf')]
    if all_bd:
        min_bd = min(all_bd)
        max_bd = max(all_bd)
        plt.plot([min_bd, max_bd], [min_bd, max_bd], 'k--', alpha=0.5)

    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title(f"Diagrama de persistencia - componente {idx}")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"persistencia_diagrama_comp{idx}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Diagrama de puntos guardado: {output_path}")


def plot_persistence_barcode_gudhi(persistence, idx, output_dir):
    plt.figure(figsize=(8, 4))

    y_offset = 0
    legend_handles = {}
    colors = {}

    dims_presentes = sorted(set(dim for dim, _ in persistence))
    for i, dim in enumerate(dims_presentes):
        colors[dim] = f"C{i}"

    for dim in dims_presentes:
        bars = [(b, d) for d_dim, (b, d) in persistence if d_dim == dim and d != float('inf')]
        for b, d in bars:
            plt.hlines(y=y_offset, xmin=b, xmax=d, colors=colors[dim], lw=2)
            y_offset += 1
        legend_handles[dim] = plt.Line2D([0], [0], color=colors[dim], lw=2, label=f"H{dim}")

    plt.xlabel("Filtración")
    plt.ylabel("Intervalos")
    plt.title(f"Barcode persistencia - componente {idx}")
    plt.legend(handles=list(legend_handles.values()))
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"persistencia_barcode_comp{idx}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Barcode guardado: {output_path}")


def calcular_persistencia_gudhi(X, puntos_indices, idx, output_dir):
    puntos = X[puntos_indices]

    if len(puntos) < 2:
        print(f"⚠️ Componente {idx}: muy pocos puntos para persistencia")
        return

    diam = pdist(puntos).max()
    print(f"   Diámetro del componente: {diam:.3f}")

    rips_complex = gd.RipsComplex(points=puntos, max_edge_length=diam * 1.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    print(f"   Número de símplices: {simplex_tree.num_simplices()}")

    if simplex_tree.num_simplices() == 0:
        print(f"⚠️ Componente {idx}: sin símplices válidos")
        return

    persistence = simplex_tree.persistence()
    persistence = [pair for pair in persistence if pair[1][1] != float('inf')]

    if len(persistence) == 0:
        print(f"⚠️ Componente {idx}: sin pares persistentes finitos")
        return

    plot_persistence_diagram_gudhi(persistence, idx, output_dir)
    plot_persistence_barcode_gudhi(persistence, idx, output_dir)


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
        and f.endswith('_estandardizado.log')
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


def calcular_persistencia_por_componente(graph, X, min_nodes=8, output_dir="tda_persistencia"):
    os.makedirs(output_dir, exist_ok=True)

    G_nx = km.to_nx(graph)
    componentes = list(nx.connected_components(G_nx))

    print(f"🔹 Se detectaron {len(componentes)} componentes conexas")

    for idx, comp in enumerate(componentes):
        puntos_indices = []
        for nodo in comp:
            puntos_indices.extend(graph['nodes'][nodo])
        puntos_indices = list(set(puntos_indices))

        print(f"➡ Componente {idx}: {len(comp)} nodos, {len(puntos_indices)} puntos")

        if len(comp) > min_nodes and len(puntos_indices) >= 20:
            calcular_persistencia_gudhi(X, puntos_indices, idx, output_dir)
        else:
            print(f"⚠️ Componente {idx}: no cumple los criterios para persistencia")


def aplicar_tda_mapper_con_persistencia(df, output_html_path):
    if df.empty:
        print("⚠️ El DataFrame está vacío. No se puede aplicar TDA.")
        return

    df_numerico = df.select_dtypes(include='number').dropna()

    # PCA 3D
    pca = PCA(n_components=3)
    pca_3d = pca.fit_transform(df_numerico.values)

    # Mapper
    mapper = km.KeplerMapper(verbose=1)

    graph = mapper.map(
        lens=pca_3d,
        X=df_numerico.values,
        cover=km.Cover(n_cubes=58, perc_overlap=0.4),
        clusterer=DBSCAN(eps=3, min_samples=13)
    )

    # Tooltips
    tooltips = df.loc[df_numerico.index, 'prefijo_origen'].astype(str).values

    # Visualización
    mapper.visualize(
        graph,
        path_html=output_html_path,
        title="TDA Mapper con persistencia (prefijo)",
        #color_values=tooltips,
        color_function_name="Prefijo",
        custom_tooltips=tooltips
    )

    print(f"✅ Grafo guardado en: {output_html_path}")

    # Calcular diagramas de persistencia para componentes grandes
    calcular_persistencia_por_componente(graph, df_numerico.values, min_nodes=8)


# === USO DE EJEMPLO ===
if __name__ == "__main__":
    # Aquí cargarías tu grafo y tus datos X
    # Ejemplo: graph = tu_grafo_mapper; X = tus_datos_originales_o_pca
    prefijo1 = 'MIA'
    prefijo2 = 'MPFC'
    prefijo3 = 'MRS'
    prefijo4 = 'MSC'
    prefijo5 = 'MSFC'
    prefijo6 = 'MVIE'
    prefijo7 = 'MVIV'
    prefijo8 = 'WIDS'
    prefijo9 = 'WIDSL'
    prefijo10 = 'WLEC'
    prefijo11 = 'WSUT'
    prefijo12 = 'WVAE'
    prefijo13 = 'WVAL'
    prefijo14 = 'WVIV'
    prefijos=[prefijo1,prefijo3,prefijo4,prefijo5,prefijo6,prefijo7,prefijo8,prefijo9,prefijo10,prefijo11,prefijo12,prefijo13,prefijo14]

    print("🚀 Este es un módulo de persistencia. Importa o llama calcular_persistencia_por_componente(graph, X)")

    output_directory = "datosEstandarizados/" # <- AJUSTA esta ruta
    lista_prefijos = prefijos

    output_html_file = "tda_mapper_pca3d_persistencia.html"

    df_union = cargar_y_unir_archivos_por_prefijos(output_directory, lista_prefijos)
    aplicar_tda_mapper_con_persistencia(df_union, output_html_file)
