import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

from utils.utils_plot import ORDEN_ALGORITMOS, guardar_dataframe


def comparar_algoritmos_por_porcentaje_version(carpetas_csv, nombres_versiones, carpeta_salida):
    """
    Compara dos o más versiones de un mismo algoritmo agrupando por porcentaje inicial.

    Args:
        carpetas_csv (list): Lista de rutas a las carpetas que contienen los CSVs.
        nombres_versiones (list): Lista de nombres a mostrar para cada versión (para la leyenda).
        carpeta_salida (str): Carpeta donde se guardará el gráfico.
    """
    if len(carpetas_csv) != len(nombres_versiones):
        raise ValueError("El número de carpetas y el número de nombres de versiones deben coincidir.")

    dfs = []

    for carpeta, nombre_version in zip(carpetas_csv, nombres_versiones):
        if not os.path.isdir(carpeta):
            print(f"Carpeta no encontrada: {carpeta}")
            continue

        archivos = [f for f in os.listdir(carpeta) if f.endswith(".csv")]
        for archivo in archivos:
            path = os.path.join(carpeta, archivo)
            df = pd.read_csv(path)
            df["Version"] = nombre_version
            dfs.append(df)

    if not dfs:
        print("No se encontraron CSVs válidos.")
        return

    df_comb = pd.concat(dfs, ignore_index=True)
    
    guardar_dataframe(df_comb, "tmp")

    if not {"Porcentaje Inicial", "Accuracy", "Version"}.issubset(df_comb.columns):
        print("Faltan columnas necesarias en los CSVs.")
        return

    df_comb["Porcentaje Inicial (%)"] = df_comb["Porcentaje Inicial"] * 100

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_comb,
        x="Porcentaje Inicial (%)",
        y="Accuracy",
        hue="Version",
        errorbar="sd",
        palette="Set2"
    )
    plt.title("Comparativa de Accuracy por Porcentaje Inicial y Versión", fontsize=13)
    plt.ylabel("Accuracy", fontsize=11.5)
    plt.xlabel("Porcentaje Inicial (%)", fontsize=11.5)
    plt.legend(title="Versión")
    plt.tight_layout()
    plt.xticks(fontsize=11.5, fontweight='bold')
    plt.yticks(fontsize=11.5, fontweight='bold')

    os.makedirs(carpeta_salida, exist_ok=True)
    output_path = os.path.join(carpeta_salida, "comparacion_por_porcentaje.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def comparar_algoritmos_por_algoritmo(carpetas_csv, nombres_versiones, nombres_algoritmos, carpeta_salida):
    """
    Compara Accuracy entre diferentes algoritmos y versiones (barplot agrupado por Algoritmo con hue=Versión).

    Args:
        carpetas_csv (list): Lista de carpetas que contienen los CSVs.
        nombres_versiones (list): Lista de nombres para cada versión.
        nombres_algoritmos (list): Lista de nombres para cada algoritmo.
        carpeta_salida (str): Carpeta donde guardar el gráfico.
    """
    if not (len(carpetas_csv) == len(nombres_versiones) == len(nombres_algoritmos)):
        raise ValueError("El número de carpetas, nombres de versiones y nombres de algoritmos debe coincidir.")

    dfs = []
    for carpeta, nombre_version, nombre_algoritmo in zip(carpetas_csv, nombres_versiones, nombres_algoritmos):
        if not os.path.isdir(carpeta):
            print(f"Carpeta no encontrada: {carpeta}")
            continue
        archivos = [f for f in os.listdir(carpeta) if f.endswith(".csv")]
        for archivo in archivos:
            path = os.path.join(carpeta, archivo)
            df = pd.read_csv(path)
            df["Version"] = nombre_version
            df["Algoritmo"] = nombre_algoritmo
            dfs.append(df)

    if not dfs:
        print("No se encontraron CSVs válidos.")
        return

    df_comb = pd.concat(dfs, ignore_index=True)
    
    guardar_dataframe(df_comb, "tmp")

    if not {"Algoritmo", "Accuracy", "Version"}.issubset(df_comb.columns):
        print("Faltan columnas necesarias en los CSVs.")
        return

    orden_algoritmos = [alg for alg in ORDEN_ALGORITMOS if alg in df_comb["Algoritmo"].unique()]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_comb,
        x="Algoritmo",
        y="Accuracy",
        hue="Version",
        order=orden_algoritmos if orden_algoritmos else sorted(df_comb["Algoritmo"].unique()),
        errorbar="sd",
        palette="Set2"
    )
    plt.title("Comparativa de Accuracy por Algoritmo y Versión", fontsize=13)
    plt.ylabel("Accuracy", fontsize=11.5)
    plt.xlabel("Algoritmo", fontsize=11.5)
    plt.xticks(fontsize=11.5, fontweight='bold')
    plt.yticks(fontsize=11.5, fontweight='bold')
    plt.legend(title="Versión", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(carpeta_salida, exist_ok=True)
    output_path = os.path.join(carpeta_salida, "comparacion_por_algoritmo.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar Accuracy entre versiones de un algoritmo")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Lista de carpetas que contienen los CSVs")
    parser.add_argument("--NOMBRES_VERSIONES", nargs='+', required=True, help="Lista de nombres de versión")
    parser.add_argument("--NOMBRES_ALGORITMOS", nargs='+', required=True, help="Lista de nombres de algoritmo (eje X)")
    parser.add_argument("--SALIDA", type=str, required=True, help="Carpeta de salida para el gráfico")
    parser.add_argument("--MODO", choices=["porcentaje", "algoritmo"], required=True, help="Modo de comparación: 'porcentaje' o 'algoritmo'")

    args = parser.parse_args()

    if args.MODO == "porcentaje":
        comparar_algoritmos_por_porcentaje_version(
            carpetas_csv=args.CARPETAS,
            nombres_versiones=args.NOMBRES_VERSIONES,
            carpeta_salida=args.SALIDA
        )
    elif args.MODO == "algoritmo":
        comparar_algoritmos_por_algoritmo(
            carpetas_csv=args.CARPETAS,
            nombres_versiones=args.NOMBRES_VERSIONES,
            nombres_algoritmos=args.NOMBRES_ALGORITMOS,
            carpeta_salida=args.SALIDA
        )
