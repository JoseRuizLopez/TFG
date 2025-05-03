import os
import re
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


def plot_min_max_lines(df: pd.DataFrame, y_col: str, x_col: str, ax: plt.Axes):
    grouped = df.groupby(x_col)[y_col].agg(['min', 'max']).reset_index()

    # Obtener lo que se ve realmente en el eje X
    xlabels = [tick.get_text() for tick in ax.get_xticklabels()]
    xticks = ax.get_xticks()
    positions = dict(zip(xlabels, xticks))

    for _, row in grouped.iterrows():
        cat = str(row[x_col])
        min_val = row['min']
        max_val = row['max']

        if cat in positions:
            xpos = positions[cat]
            ax.hlines(min_val, xpos - 0.2, xpos + 0.2, color='red', lw=2)
            ax.hlines(max_val, xpos - 0.2, xpos + 0.2, color='blue', lw=2)
            ax.text(xpos, min_val, f'{min_val:.3f}', ha='center', va='top', fontsize=9, color='red')
            ax.text(xpos, max_val, f'{max_val:.3f}', ha='center', va='bottom', fontsize=9, color='blue')


def plot_boxplot(df: pd.DataFrame, metric: str, filename: str | None, hue: str | None, title: str, eje_x: str):
    """
    Genera un boxplot personalizado con orden automático de categorías y min/max destacados.

    Args:
        df: DataFrame con los datos.
        metric: Columna a usar como métrica en el eje Y.
        filename: Archivo de salida (opcional).
        hue: Columna a usar como color (opcional).
        title: Título del gráfico.
        eje_x: Columna a usar como eje X.
    """
    if metric.title() not in df.columns or eje_x not in df.columns:
        raise ValueError(f"Faltan columnas necesarias: '{metric.title()}' o '{eje_x}'.")

    # Determinar orden personalizado del eje X (alfabético o numérico si se puede)
    categorias = df[eje_x].dropna().unique()
    try:
        # Si se pueden convertir a int, orden numérico
        orden_x = sorted(categorias, key=lambda x: int(x))
    except ValueError:
        orden_x = sorted(categorias)

    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x=eje_x, y=metric.title(), order=orden_x)

    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.xlabel(eje_x)
    plt.ylabel(metric.title())

    if hue and hue in df.columns:
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='best')

    plot_min_max_lines(df, metric.title(), eje_x, ax)

    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)

    
def sort_natural(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def plot_porcentajes_por_algoritmo(
    df: pd.DataFrame,
    tipo: str,
    columnas_clase: list | None = None,
    filename: str | None = None,
    modo: str = "juntos"
):
    """
    Genera un gráfico de barras comparando porcentajes agrupados por algoritmo.

    Args:
        df: DataFrame con los datos.
        tipo: "inicial_final" o "clases".
        columnas_clase: Columnas de clases (solo para tipo="clases").
        filename: Ruta del archivo a guardar (opcional).
        modo: "libres", "no_libres", "ambos", "juntos"
    """
    if "Algoritmo" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Algoritmo'.")

    if tipo == "inicial_final":
        columnas_necesarias = {"Porcentaje Inicial", "Porcentaje Final"}
        if not columnas_necesarias.issubset(df.columns):
            raise ValueError(f"Faltan columnas necesarias: {columnas_necesarias - set(df.columns)}")

        def preparar(df_sub):
            return df_sub.melt(id_vars=["Algoritmo"],
                               value_vars=["Porcentaje Inicial", "Porcentaje Final"],
                               var_name="Tipo", value_name="Porcentaje"), "Tipo", "Porcentaje Inicial vs Final por Algoritmo"

    elif tipo == "clases":
        if not columnas_clase:
            raise ValueError("Debes proporcionar 'columnas_clase' cuando tipo='clases'.")
        def preparar(df_sub):
            return df_sub.melt(id_vars=["Algoritmo"],
                               value_vars=columnas_clase,
                               var_name="Clase", value_name="Porcentaje"), "Clase", "Distribución de Clases por Algoritmo"
    else:
        raise ValueError("El parámetro 'tipo' debe ser 'inicial_final' o 'clases'.")

    # Separación por modo
    df_libres = df[df["Algoritmo"].str.contains("libre", case=False)]
    df_no_libres = df[~df["Algoritmo"].str.contains("libre", case=False)]

    if modo not in {"libres", "no_libres", "ambos", "juntos"}:
        raise ValueError("El parámetro 'modo' debe ser 'libres', 'no_libres', 'ambos' o 'juntos'.")

    if modo in {"libres", "no_libres"}:
        subset_df = df_libres if modo == "libres" else df_no_libres

        if subset_df.empty:
            print(f"No hay datos para el modo '{modo}'.")
            return

        df_melt, hue_col, titulo = preparar(subset_df)
        orden_algoritmos = sorted(df_melt["Algoritmo"].unique(), key=sort_natural)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Algoritmo", y="Porcentaje", hue=hue_col,
                    order=orden_algoritmos, errorbar=None)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{titulo} - Algoritmos {modo.replace('_', ' ').title()}")
        plt.ylabel("Porcentaje (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif modo == "ambos":
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for subset_df, title, ax in [
            (df_libres, "Algoritmos Libres", axs[0]),
            (df_no_libres, "Algoritmos No Libres", axs[1])
        ]:
            if subset_df.empty:
                ax.set_visible(False)
                continue

            df_melt, hue_col, titulo = preparar(subset_df)
            orden_algoritmos = sorted(df_melt["Algoritmo"].unique(), key=sort_natural)

            sns.barplot(data=df_melt, x="Algoritmo", y="Porcentaje", hue=hue_col,
                        order=orden_algoritmos, errorbar=None, ax=ax)
            ax.set_title(f"{titulo} - {title}")
            ax.set_xlabel("Algoritmo")
            ax.set_ylabel("Porcentaje (%)")
            ax.tick_params(axis='x', rotation=45)
            ax.legend(loc='upper right')

        plt.tight_layout()

    elif modo == "juntos":
        df_melt, hue_col, titulo = preparar(df)
        orden_algoritmos = sorted(df_melt["Algoritmo"].unique(), key=sort_natural)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x="Algoritmo", y="Porcentaje", hue=hue_col,
                    order=orden_algoritmos, errorbar=None)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{titulo} - Todos los algoritmos juntos")
        plt.ylabel("Porcentaje (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Gráfico guardado en {filename}.")


def plot_porcentajes_por_porcentaje_inicial(df: pd.DataFrame, filename: str | None = None, modo: str = "ambos"):
    """
    Genera un gráfico de barras comparando Porcentaje Inicial vs Final,
    agrupado por Porcentaje Inicial. Puede separarse por tipo de algoritmo o mostrarse todo junto.

    Args:
        df: DataFrame con los datos.
        filename: Nombre del archivo de salida (opcional).
        modo: "libres", "no_libres", "ambos" o "juntos".
    """
    required_columns = {"Porcentaje Inicial", "Porcentaje Final", "Algoritmo"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Faltan columnas necesarias: {', '.join(missing)}")

    df["Porcentaje Inicial"] = df["Porcentaje Inicial"].astype(str)

    if modo not in {"libres", "no_libres", "ambos", "juntos"}:
        raise ValueError("El parámetro 'modo' debe ser 'libres', 'no_libres', 'ambos' o 'juntos'.")

    df_libres = df[df["Algoritmo"].str.contains("libre", case=False)]
    df_no_libres = df[~df["Algoritmo"].str.contains("libre", case=False)]

    if modo in {"libres", "no_libres"}:
        subset_df = df_libres if modo == "libres" else df_no_libres

        if subset_df.empty:
            print(f"No hay datos para el modo '{modo}'.")
            return

        df_melt = subset_df.melt(
            id_vars=["Porcentaje Inicial"],
            value_vars=["Porcentaje Inicial", "Porcentaje Final"],
            var_name="Tipo", value_name="Porcentaje"
        )
        orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: int(x))

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_melt,
            x="Porcentaje Inicial",
            y="Porcentaje",
            hue="Tipo",
            order=orden_x,
            errorbar=None
        )
        plt.title(f"Porcentaje Inicial vs Final - Algoritmos {modo.replace('_', ' ').title()}")
        plt.xlabel("Porcentaje Inicial")
        plt.ylabel("Porcentaje (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    elif modo == "ambos":
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for subset_df, title, ax in [
            (df_libres, "Algoritmos Libres", axs[0]),
            (df_no_libres, "Algoritmos No Libres", axs[1])
        ]:
            if subset_df.empty:
                ax.set_visible(False)
                continue

            df_melt = subset_df.melt(
                id_vars=["Porcentaje Inicial"],
                value_vars=["Porcentaje Inicial", "Porcentaje Final"],
                var_name="Tipo", value_name="Porcentaje"
            )
            orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: float(x))


            sns.barplot(
                data=df_melt,
                x="Porcentaje Inicial",
                y="Porcentaje",
                hue="Tipo",
                order=orden_x,
                errorbar=None,
                ax=ax
            )
            ax.set_title(title)
            ax.set_xlabel("Porcentaje Inicial")
            ax.set_ylabel("Porcentaje (%)")
            ax.tick_params(axis='x', rotation=45)
            ax.legend(loc='upper right')

        plt.tight_layout()

    elif modo == "juntos":
        df_melt = df.melt(
            id_vars=["Porcentaje Inicial"],
            value_vars=["Porcentaje Inicial", "Porcentaje Final"],
            var_name="Tipo", value_name="Porcentaje"
        )
        orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: int(x))

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_melt,
            x="Porcentaje Inicial",
            y="Porcentaje",
            hue="Tipo",
            order=orden_x,
            errorbar=None
        )
        plt.title("Porcentaje Inicial vs Final - Todos los algoritmos juntos")
        plt.xlabel("Porcentaje Inicial")
        plt.ylabel("Porcentaje (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Gráfico guardado en {filename}.")


def generate_plots_from_csvs(
    archivos_csv=[
        "results/csvs/resultados_2025-02-23_17-06_task_-1.csv"
    ],
    carpeta_img: str | None = None,
    modelo_name: str | None = None,
    carpeta_csv: str | None = None,
):
    path_csvs = os.path.dirname(archivos_csv[0]) if carpeta_csv is None else carpeta_csv
    dataframes = []

    columnas_numericas = ["Accuracy", "Precision", "Recall", "F1-score",
                          "Evaluaciones", "Porcentaje"]

    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        for col in df.columns:
            if col.split(' ')[0] in columnas_numericas:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Duracion'] = pd.to_timedelta(df['Duracion'], errors='coerce')
        dataframes.append(df)

    # Guardar todos los DataFrames
    os.makedirs(path_csvs, exist_ok=True)
    
    df_concatenado = pd.concat(dataframes)
    df_concatenado.to_csv(f'{path_csvs}/concatenado.csv', index=True)
    

    # Convertir Duracion a segundos para incluirla en el promedio
    df_modified = df_concatenado.copy()
    df_modified["Duracion_segundos"] = df_modified["Duracion"].dt.total_seconds()

    # Agrupar y calcular media incluyendo la duración
    agrupado_media = df_modified.groupby(["Algoritmo", "Porcentaje Inicial"], as_index=False).mean(numeric_only=True)

    # Convertir duración promedio de nuevo a formato timedelta
    agrupado_media["Duracion"] = pd.to_timedelta(agrupado_media["Duracion_segundos"], unit='s')
    agrupado_media.drop(columns=["Duracion_segundos"], inplace=True)

    agrupado_media.to_csv(f"{carpeta_csv}/media.csv", index=False)

    agrupado_mejor = df_concatenado.sort_values(by="Accuracy", ascending=False).groupby(["Algoritmo", "Porcentaje Inicial"], as_index=False).first()
    agrupado_mejor.to_csv(f"{carpeta_csv}/mejor.csv", index=False)


    df = pd.read_csv(f'{path_csvs}/concatenado.csv')

    if modelo_name is None:
        modelo_name = ""

    if carpeta_img is not None:
        filename1 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-porcentaje.png'
        filename2 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-algoritmo.png'
        filename3 = f'{carpeta_img}/{modelo_name}-BARPLOT-balance-de-clases-por-algoritmo.png'
        filename4 = f'{carpeta_img}/{modelo_name}-BARPLOT-porcentaje-inical-vs-final-por-algoritmo.png'
        filename5 = f'{carpeta_img}/{modelo_name}-BARPLOT-porcentaje-inicial-vs-final-por-pi.png'
    else:
        filename1 = filename2 = filename3 = filename4 = filename5 = None


    # ====== Boxplot 1: Accuracy vs Porcentaje Inicial ======
    plot_boxplot(
        df=df,
        metric="Accuracy",
        eje_x="Porcentaje Inicial",
        hue=None,
        title="Comparación de Accuracy según Porcentaje Inicial y Algoritmo",
        filename=filename1
    )

    # ====== Boxplot 2: Accuracy vs Algoritmo ======
    plot_boxplot(
        df=df,
        metric="Accuracy",
        eje_x="Algoritmo",
        hue=None,
        title="Comparación de Accuracy según Algoritmo y Porcentaje Inicial",
        filename=filename2
    )

    if filename1 and filename2:
        print(f"Los Boxplot se han guardado en {filename1} y {filename2}.")
        

    columnas_clase = [col for col in df.columns if col.startswith("Porcentaje ") and col not in ["Porcentaje Inicial", "Porcentaje Final"]]
    
    if "Porcentaje Final" in df.columns and len(columnas_clase) > 1:
        plot_porcentajes_por_algoritmo(df, tipo="clases", filename=filename3, columnas_clase=columnas_clase)
        plot_porcentajes_por_algoritmo(df, tipo="inicial_final", filename=filename4, modo="ambos")
        plot_porcentajes_por_porcentaje_inicial(df, filename=filename5, modo="ambos")
        
        print("Se han generado los diagramas de barras.")
