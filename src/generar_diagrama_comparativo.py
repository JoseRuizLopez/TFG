import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

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
            print(f"[AVISO] Carpeta no encontrada: {carpeta}")
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
    plt.title("Comparativa de Accuracy por Porcentaje Inicial y Versión")
    plt.ylabel("Accuracy")
    plt.xlabel("Porcentaje Inicial (%)")
    plt.legend(title="Versión")
    plt.tight_layout()

    os.makedirs(carpeta_salida, exist_ok=True)
    output_path = os.path.join(carpeta_salida, "comparacion_por_porcentaje.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar Accuracy por Porcentaje Inicial entre versiones de un mismo algoritmo")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Lista de carpetas que contienen los CSVs")
    parser.add_argument("--NOMBRES", nargs='+', required=True, help="Lista de nombres a mostrar para cada versión")
    parser.add_argument("--SALIDA", type=str, required=True, help="Carpeta de salida para el gráfico")

    args = parser.parse_args()

    comparar_algoritmos_por_porcentaje_version(
        carpetas_csv=args.CARPETAS,
        nombres_versiones=args.NOMBRES,
        carpeta_salida=args.SALIDA
    )
