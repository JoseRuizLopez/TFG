import argparse
import os
import pandas as pd

from utils.utils_plot import plot_boxplot


def comparar_versiones(output_path, carpetas_versiones, nombres_versiones, modelo=None, metricas=None, titulo_base=None):
    carpeta_salida_img = os.path.join("img/finales", output_path)
    os.makedirs(carpeta_salida_img, exist_ok=True)

    if len(carpetas_versiones) != len(nombres_versiones):
        print("El número de rutas debe coincidir con el número de nombres de versiones.")
        return

    dfs = []

    for carpeta_relativa, nombre_version in zip(carpetas_versiones, nombres_versiones):
        ruta_carpeta = os.path.join("results/csvs", carpeta_relativa)
        if not os.path.isdir(ruta_carpeta):
            print(f"Carpeta no encontrada: {ruta_carpeta}")
            continue

        for archivo in os.listdir(ruta_carpeta):
            if archivo.endswith(".csv"):
                df = pd.read_csv(os.path.join(ruta_carpeta, archivo))
                df["Algoritmo"] = nombre_version
                if modelo and "Modelo" in df.columns:
                    df = df[df["Modelo"] == modelo]
                dfs.append(df)

    if not dfs:
        print("No se encontraron datos en las carpetas especificadas.")
        return
    
    df_total = pd.concat(dfs, ignore_index=True)

    metricas_a_graficar = metricas or ["Accuracy"]
    for metrica in metricas_a_graficar:
        titulo = f"{titulo_base}" if titulo_base else f"Comparación de {metrica}"
        output_file = os.path.join(carpeta_salida_img, f"comparacion_{metrica}.png")
        plot_boxplot(
            df=df_total,
            metric=metrica,
            filename=output_file,
            hue=None,
            title=titulo,
            eje_x="Algoritmo"
        )
        print(f"Guardado: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar versiones de resultados en carpetas directas")
    parser.add_argument("--OUT", type=str, required=True, help="Nombre de la carpeta de salida (ej: fecha)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Rutas de carpetas de versiones (ej: comparaciones/AM comparaciones/WC)")
    parser.add_argument("--NOMBRES", nargs='+', required=True, help="Nombres que aparecerán en el gráfico (ej: AM WC)")
    parser.add_argument("--MODELO", type=str, required=False, help="Filtrar por modelo (opcional)")
    parser.add_argument("--METRICAS", nargs='*', help="Lista de métricas a comparar (ej: Accuracy Precision Recall)")
    parser.add_argument("--TITULO", type=str, required=False, help="Título base para los gráficos (opcional)")

    args = parser.parse_args()

    comparar_versiones(
        output_path=args.OUT,
        carpetas_versiones=args.CARPETAS,
        nombres_versiones=args.NOMBRES,
        modelo=args.MODELO,
        metricas=args.METRICAS,
        titulo_base=args.TITULO
    )
