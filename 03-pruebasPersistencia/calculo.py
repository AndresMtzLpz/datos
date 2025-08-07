import os
import numpy as np
import matplotlib.pyplot as plt
from gudhi.cover_complex import MapperComplex, GraphInducedComplex, NerveComplex
from gudhi import bottleneck_distance
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import gudhi as gd
import pandas as pd
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
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

if __name__ == "__main__":
    prefijo0 = 'MCEFL'
    prefijo1 = 'MIA'
    prefijo2 = 'MPFC' #
    prefijo3 = 'MRS'
    prefijo4 = 'MSC'
    prefijo5 = 'MSFC'
    prefijo6 = 'MVIE' #
    prefijo7 = 'MVIV'
    prefijo8 = 'WIDS'
    prefijo9 = 'WIDSL'
    prefijo10 = 'WLEC'
    prefijo11 = 'WSUT'
    prefijo12 = 'WVAE'
    prefijo13 = 'WVAL'
    prefijo14 = 'WVIV'
    #prefijos=[prefijo0,prefijo1,prefijo3,prefijo4,prefijo5,prefijo7,prefijo8,prefijo9,prefijo10,prefijo11,prefijo12,prefijo13,prefijo14]
    #prefijos = ['MIA', 'MRS', 'MSC', 'MSFC', 'MVIE', 'MVIV', 'WIDS', 'WIDSL', 'WLEC', 'WSUT', 'WVAE', 'WVAL', 'WVIV']
    prefijos=[prefijo1]

    
    output_directory = "datosEstandarizados3m_90/"  # <-- AJUSTA esta ruta

    #Cargar los archivos correspondientes al prefijo
    df_union = cargar_y_unir_archivos_por_prefijos(output_directory, prefijos)

    #Dejar solo los valores númericos (Se quitan los agregados que ayudaron a clasificar por prefijo)
    df_numerico = df_union.select_dtypes(include='number').dropna()

    #Obtener matriz de distancia
    D = pairwise_distances(df_numerico.values)

    #Se establece el cover - Este no cambia, se usa como motor    
    cover_complex = MapperComplex(
        input_type='distance matrix', min_points_per_node=0,
        clustering=None, N=1000, beta=0., C=5000,
        filter_bnds=None, resolutions=[40,3], gains=None, verbose=verbose)

    print("Se construye complejidad")
    #Se construye la omplejidad de Mapper dentro de cover_complex
    _ = cover_complex.fit(D)

    #Se obtiene el grafo con networkx
    G = cover_complex.get_networkx()

    #Se muestra el grafo
    plt.figure()
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=[cover_complex.node_info_[v]["colors"][0] for v in G.nodes()])
    plt.show()

    #Se guarda en html
    cover_complex.save_to_html(file_name="human", data_name="human", cover_name="uniform", color_name="height")

    cover_complex.data = df_numerico.values

    print("Se obtiene grafo networkx")
    #Se obtiene el grafo con networkx
    G = cover_complex.get_networkx()

    #Se muestra el grafo
    plt.figure()
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_color=[cover_complex.node_info_[v]["colors"][0] for v in G.nodes()])
    plt.show()

    print("Se guarda html")
    #Se guarda en html
    cover_complex.save_to_html(file_name="human", data_name="human", cover_name="uniform", color_name="height")

    cover_complex.data = df_numerico.values

    
    connected_components = list(nx.connected_components(G))
    print(f"Se obtienen componentes conexas: {connected_components}")

    print("Se hace el calculo de persistencia")

    for i, component_nodes in enumerate(connected_components):
        # Get all point indices for this component by combining all points from its nodes
        all_point_indices = []
        for node_label in component_nodes:
            point_indices = [cover_complex.node_info_[v]['colors'][0] for v in G.nodes()]
            all_point_indices.extend(point_indices)
        
        component_points = df_numerico.iloc[np.unique(all_point_indices)]

        #print(component_points)
        
        print(f"Component {i+1} has {len(component_points)} total points from {len(component_nodes)} nodes.")

        # Compute Persistence on the combined point cloud of the component
        rips_complex = gd.RipsComplex(points=component_points.values, max_edge_length=3.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence = simplex_tree.persistence()
        

        # Generar el diagrama (AxesSubplot)
        ax = gd.plot_persistence_diagram(persistence)

        # Obtener la figura asociada al Axes
        fig = ax.get_figure()

        # Guardar como PNG
        fig.savefig(f"diagrama_persistencia{i}.png", dpi=300, bbox_inches='tight')

        # (Opcional) Cierra la figura para liberar memoria
        plt.close(fig)
