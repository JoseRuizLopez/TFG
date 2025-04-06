import os
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def plot_fitness_evolution(
    fitness_history: List[float],
    initial_percentage: int,
    algorithm_name: str,
    metric: str,
    model: str,
    carpeta: str
):
    """
    Crea y guarda una gráfica que muestra la evolución del fitness.

    Args:
        fitness_history: Lista con los valores de fitness
        initial_percentage: Entero con el porcentaje inicial de imagenes seleccionadas
        algorithm_name: Nombre del algoritmo utilizado
        metric: Métrica utilizada (accuracy o f1)
        model: Nombre del modelo usado
        carpeta: Nombre de la carpeta
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o')
    plt.title(f'Evolución del {metric} - Algoritmo {algorithm_name} - Modelo {model} - '
              f'Porcentaje Inicial {str(initial_percentage)}%')
    plt.xlabel('Iteración')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.savefig(f'{carpeta}/{model}-{algorithm_name.replace(" ", "_")}-{str(initial_percentage)}-{metric}.png')
    plt.close()


def plot_multiple_fitness_evolution(
    data: List[List[float]],
    labels: List[str],
    metric: str,
    title: str,
    filename: str,
    x_label: str = "Iteración"
):
    """
    Crea y guarda una gráfica que muestra la evolución del fitness multiple.

    Args:
        data: Lista de listas, donde cada lista interna contiene los valores de una línea a lo largo de las iteraciones
        labels: Lista de etiquetas para cada línea
        metric: Métrica utilizada (accuracy o f1)
        title: Título de la gráfica
        filename: Nombre del archivo generado en el directorio
        x_label: Nombre de la etiqueta para el eje x
    """
    # Encontrar la longitud máxima entre todas las listas
    max_length = max(len(lst) for lst in data)
    max_length = max_length if max_length != 1 else 50

    # Extender cada lista repitiendo el último valor hasta la longitud máxima
    extended_data = []
    for lst in data:
        extended_lst = lst + [lst[-1]] * (max_length - len(lst))
        extended_data.append(extended_lst)

    plt.figure(figsize=(10, 6))

    # Graficar cada lista interna de data
    for i, line_data in enumerate(extended_data):
        plt.plot(line_data, label=labels[i])

    # Títulos y etiquetas
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)

    # Mostrar la leyenda
    plt.legend(loc='best', fontsize=10)

    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_min_max_lines(df: pd.DataFrame | pl.DataFrame, y_col, x_col):
    # Convertir a pandas si es un DataFrame de polars (porque matplotlib funciona mejor con pandas)
    if isinstance(df, pl.DataFrame):
        # df = df.to_pandas()  # No disponible por la version de polars
        df = pd.DataFrame(df.to_dict())

    # Obtener los valores mínimo y máximo por cada categoría en x_col
    grouped = df.groupby(x_col)[y_col].agg(['min', 'max'])

    # Iterar sobre cada categoría
    for i, (cat, values) in enumerate(grouped.iterrows()):
        min_val, max_val = values['min'], values['max']

        # Dibujar líneas horizontales en los valores min y max
        plt.plot([i - 0.2, i + 0.2], [min_val, min_val], color='red', lw=2)
        plt.plot([i - 0.2, i + 0.2], [max_val, max_val], color='blue', lw=2)

        # Anotar los valores exactos
        plt.text(i, min_val, f'{min_val:.3f}', ha='center', va='top', fontsize=10, color='red')
        plt.text(i, max_val, f'{max_val:.3f}', ha='center', va='bottom', fontsize=10, color='blue')


def plot_boxplot(df: pd.DataFrame, metric: str, filename: str | None, hue: str | None, title: str, eje_x: str):

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x=eje_x, y=metric.title())

    plt.title(title)
    plt.xlabel(eje_x)
    plt.ylabel(metric.title())
    if hue:
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='best')

    plot_min_max_lines(df, metric, eje_x)

    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def generate_boxplot_from_csvs(
    # Lista de archivos CSV
    archivos_csv=[
        "results/csvs/resultados_2025-02-23_17-06_task_-1.csv"
    ],
    carpeta_img: str | None = None,
    modelo_name: str | None = None,
):
    path_media = os.path.dirname(archivos_csv[0])

    # Lista para almacenar cada DataFrame
    dataframes = []

    # Especificar las columnas que deberían ser numéricas
    columnas_numericas = ["Accuracy", "Precision", "Recall", "F1-score",
                          "Evaluaciones", "Porcentaje"]

    # Cargar cada archivo CSV y forzar las columnas numéricas
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)

        for col in df.columns:
            if col.split(' ')[0] in columnas_numericas:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Duracion'] = pd.to_timedelta(df['Duracion'])

        dataframes.append(df)

    # Concatenar todos los DataFrames
    df_concatenado = pd.concat(dataframes)

    # Calcular la media solo de las columnas especificadas
    df_media = df_concatenado.groupby(["Algoritmo", "Porcentaje Inicial"]).mean()

    df_media.to_csv(f'{path_media}/media.csv', index=True)

    df = pd.read_csv(f'{path_media}/media.csv')

    if modelo_name is None:
        modelo_name = ""

    if carpeta_img is not None:
        filename1 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-porcentaje.png'
        filename2 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-algoritmo.png'
    else:
        filename1 = None
        filename2 = None

    # ====== Boxplot 1: Fijando Algoritmo, variando Porcentaje Inicial ======
    plot_boxplot(
        df=df,
        metric="Accuracy",  # O "Precision"
        eje_x="Porcentaje Inicial",
        hue=None,
        title="Comparación de Accuracy según Porcentaje Inicial y Algoritmo",
        filename=filename1
    )

    # ====== Boxplot 2: Fijando Porcentaje Inicial, variando Algoritmo ======
    plot_boxplot(
        df=df,
        metric="Accuracy",  # O "Precision"
        eje_x="Algoritmo",
        hue=None,
        title="Comparación de Accuracy según Algoritmo y Porcentaje Inicial",
        filename=filename2
    )

    print(
        f"Los Boxplot se han guardado en {filename1} y {filename2}."
    )
