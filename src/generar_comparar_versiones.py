import argparse
import os
import pandas as pd

from utils.utils_plot import guardar_dataframe, plot_boxplot


def comparar_versiones(
    output_path,
    carpetas_versiones,
    nombres_algoritmos,
    nombres_versiones=None,
    metricas=None,
    titulo_base=None
):
    """
    Compara versiones y algoritmos de resultados en carpetas directas.

    Args:
        output_path (str): Nombre de la carpeta de salida para los gráficos.
        carpetas_versiones (list): Lista de rutas relativas a las carpetas de versiones.
        nombres_algoritmos (list): Lista de nombres de los algoritmos correspondientes.
        nombres_versiones (list, optional): Lista de nombres de las versiones. Si no se proporciona, se asume que no hay versiones.
        metricas (list, optional): Lista de métricas a comparar. Si no se proporciona, se usa "Accuracy".
        titulo_base (str, optional): Título base para los gráficos. Si no se proporciona, se genera un título por defecto.
    """
    carpeta_salida_img = os.path.join("img/finales", output_path)
    os.makedirs(carpeta_salida_img, exist_ok=True)

    usar_version = nombres_versiones is not None

    if not (len(carpetas_versiones) == len(nombres_algoritmos)) or (usar_version and len(carpetas_versiones) != len(nombres_versiones)):
        print("El número de rutas debe coincidir con el de algoritmos, y con versiones si se especifican.")
        return

    dfs = []

    for i, (carpeta_relativa, nombre_algoritmo) in enumerate(zip(carpetas_versiones, nombres_algoritmos)):
        ruta_carpeta = os.path.join("results/csvs", carpeta_relativa)
        if not os.path.isdir(ruta_carpeta):
            print(f"Carpeta no encontrada: {ruta_carpeta}")
            continue

        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith(".csv"):
                df = pd.read_csv(os.path.join(ruta_carpeta, archivo))
                # Forzar el nombre del algoritmo según el argumento
                df["Algoritmo"] = nombre_algoritmo
                if usar_version:
                    df["Version"] = nombres_versiones[i]
                dfs.append(df)

    if not dfs:
        print("No se encontraron datos en las carpetas especificadas.")
        return

    df_total = pd.concat(dfs, ignore_index=True)
    guardar_dataframe(df_total, "tmp")

    metricas_a_graficar = metricas or ["Accuracy"]
    for metrica in metricas_a_graficar:
        titulo = titulo_base if titulo_base else f"Comparación de {metrica}"
        output_file = os.path.join(carpeta_salida_img, f"comparacion_{metrica}.png")
        plot_boxplot(
            df=df_total,
            metric=metrica,
            filename=output_file,
            hue="Version" if usar_version else None,
            title=titulo,
            eje_x="Algoritmo",
            max_min=not usar_version
        )
        print(f"Guardado: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar versiones y algoritmos de resultados en carpetas directas")
    parser.add_argument("--OUT", type=str, required=True, help="Nombre de la carpeta de salida (ej: fecha)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Rutas de carpetas de versiones (ej: comparaciones/AM comparaciones/WC)")
    parser.add_argument("--NOMBRES_VERSIONES", nargs='+', required=False, help="Nombres de las versiones (opcional)")
    parser.add_argument("--NOMBRES_ALGORITMOS", nargs='+', required=True, help="Nombres de los algoritmos (ej: LR PR)")
    parser.add_argument("--METRICAS", nargs='*', help="Lista de métricas a comparar (ej: Accuracy Precision Recall)")
    parser.add_argument("--TITULO", type=str, required=False, help="Título base para los gráficos (opcional)")

    args = parser.parse_args()

    comparar_versiones(
        output_path=args.OUT,
        carpetas_versiones=args.CARPETAS,
        nombres_versiones=args.NOMBRES_VERSIONES,
        nombres_algoritmos=args.NOMBRES_ALGORITMOS,
        metricas=args.METRICAS,
        titulo_base=args.TITULO
    )
