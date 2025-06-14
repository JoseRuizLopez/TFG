import os
import re
from typing import List
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classes import PrintMode



ORDEN_ALGORITMOS = ["100%", "RS", "LS", "LS-F", "GA", "GA-WC", "GA-WC-F", "GA-AM", "GA-AM-F", "GA-PR", "MA", "MA-F"]


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
            ax.text(xpos, min_val - 0.0003, f'{min_val:.3f}', ha='center', va='top', fontsize=11, color='red')
            ax.text(xpos, max_val, f'{max_val:.3f}', ha='center', va='bottom', fontsize=11, color='blue')


def plot_boxplot(df: pd.DataFrame, metric: str, filename: str | None, hue: str | None, title: str, eje_x: str, max_min=True):
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
        orden_x = sorted(categorias, key=lambda x: float(x))
    except ValueError:
        orden_x = [alg for alg in ORDEN_ALGORITMOS if alg in categorias]

    plt.figure(figsize=(12, 6))
    if hue and hue in df.columns:
        ax = sns.boxplot(data=df, x=eje_x, y=metric.title(), hue=hue, order=orden_x)
    else:
        ax = sns.boxplot(data=df, x=eje_x, y=metric.title(), order=orden_x)

    plt.title(title,  fontsize=13)
    plt.xlabel(eje_x, fontsize=11.5)
    plt.ylabel(metric.title(), fontsize=11.5)

    plt.tick_params(axis='x', labelsize=11.5)
    plt.tick_params(axis='y', labelsize=11.5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    if hue and hue in df.columns:
        plt.legend(
            title=hue,
            bbox_to_anchor=(1.05, 1),
            loc='best',
            fontsize=12,            # Tamaño de la fuente de las etiquetas
            title_fontsize=13,      # Tamaño del título de la leyenda
            markerscale=1.5         # Tamaño de los iconos de color
        )

    if max_min:
        plot_min_max_lines(df, metric.title(), eje_x, ax)

    plt.grid(True)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)


def plot_barplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    order: list | None = None,
    palette: str = "Set2",
    errorbar: str | None = "sd",
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    filename: str | None = None,
    figsize: tuple = (12, 6),
    mostrar_valores: bool = False
):
    """
    Genera un barplot personalizado con orden opcional y guardado.

    Args:
        df: DataFrame con los datos.
        x: Columna para eje X.
        y: Columna para eje Y.
        hue: Columna para hue.
        order: Lista con orden de categorías en el eje X.
        palette: Paleta de colores.
        errorbar: Tipo de error (ej. 'sd', None).
        title: Título del gráfico.
        xlabel: Etiqueta del eje X.
        ylabel: Etiqueta del eje Y.
        filename: Ruta de archivo para guardado.
        figsize: Tamaño de la figura.
        mostrar_valores: Si True, se muestran los valores encima de las barras.
    """
    # Validación de columnas
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columnas faltantes: '{x}' o '{y}'.")

    # Determinar orden si no se proporciona
    if order is None:
        categorias = df[x].dropna().unique()
        try:
            order = sorted(categorias, key=lambda v: float(v))
        except (ValueError, TypeError):
            order = list(categorias)

    # Crear figura
    plt.figure(figsize=figsize)

    # Dibujar barplot
    if hue and hue in df.columns:
        ax = sns.barplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            order=order,
            palette=palette,
            errorbar=errorbar
        )
    else:
        ax = sns.barplot(
            data=df,
            x=x,
            y=y,
            order=order,
            palette=palette,
            errorbar=errorbar
        )

    if mostrar_valores:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=11.5, fontweight='bold', label_type="edge", padding=2)

    if title:
        plt.title(title, fontsize=13)
    plt.xlabel(xlabel or x, fontsize=11.5)
    plt.ylabel(ylabel or y, fontsize=11.5)
    plt.xticks(fontsize=11.5, fontweight='bold')
    plt.yticks(fontsize=11.5, fontweight='bold')
    plt.tight_layout()

    # Guardar archivo si se indica
    if filename:
        plt.savefig(filename)
    plt.close()


def plot_scatter_inicial_final(
    df: pd.DataFrame,
    filename: str | None = None,
    title: str = "Porcentaje Inicial vs Final, con líneas guía y coloreado por Accuracy",
    xlabel: str = "Porcentaje Inicial",
    ylabel: str = "Porcentaje Final",
    xticks_list: list = [0.1, 0.25, 0.5, 0.75],
    yticks_list: list = [0.1, 0.25, 0.5, 0.75],
    xlim_range: tuple = (0, 0.8),
    ylim_range: tuple = (0, 0.8)
):
    """
    Genera un scatterplot personalizado entre Porcentaje Inicial y Porcentaje Final, con líneas guía y colores por Accuracy.

    Args:
        df: DataFrame con las columnas 'Porcentaje Inicial', 'Porcentaje Final' y 'Accuracy'.
        filename: Ruta del archivo para guardar el gráfico (opcional).
        title: Título del gráfico.
        xlabel: Etiqueta del eje X.
        ylabel: Etiqueta del eje Y.
        xticks_list: Lista personalizada de ticks para el eje X.
        yticks_list: Lista personalizada de ticks para el eje Y.
        xlim_range: Rango de límites del eje X.
        ylim_range: Rango de límites del eje Y.
    """
    if not {"Porcentaje Inicial", "Porcentaje Final", "Accuracy"}.issubset(df.columns):
        raise ValueError("El DataFrame debe contener las columnas 'Porcentaje Inicial', 'Porcentaje Final' y 'Accuracy'.")

    # Crear figura
    plt.figure(figsize=(10,6))
    ax = sns.scatterplot(
        data=df,
        x="Porcentaje Inicial",
        y="Porcentaje Final",
        size="Accuracy",
        hue="Accuracy",
        sizes=(50, 300),
        palette="coolwarm",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5
    )

    # Líneas guía por cada punto
    for _, row in df.iterrows():
        plt.vlines(
            x=row["Porcentaje Inicial"],
            ymin=0,
            ymax=row["Porcentaje Final"],
            color="dimgray",
            alpha=0.8,
            linestyle="--",
            linewidth=1.5
        )
        plt.hlines(
            y=row["Porcentaje Final"],
            xmin=0,
            xmax=row["Porcentaje Inicial"],
            color="dimgray",
            alpha=0.8,
            linestyle="--",
            linewidth=1.5
        )

    # Personalización ejes
    plt.xticks(xticks_list, fontsize=11.5, fontweight='bold')
    plt.yticks(yticks_list, fontsize=11.5, fontweight='bold')
    plt.xlim(*xlim_range)
    plt.ylim(*ylim_range)

    # Títulos y etiquetas
    plt.title(title, fontsize=13)
    plt.xlabel(xlabel, fontsize=11.5)
    plt.ylabel(ylabel, fontsize=11.5)

    # Layout
    plt.tight_layout()

    # Guardar si es necesario
    if filename:
        plt.savefig(filename)
        print(f"Scatter Plot guardado en: {filename}")

    plt.close()


def sort_natural(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def plot_lineplot_accuracy_por_algoritmo(
    df: pd.DataFrame,
    filename: str | None = None,
    title: str = "Accuracy promedio por Porcentaje Inicial para cada Algoritmo"
):
    """
    Genera un gráfico de líneas con Accuracy en función del Porcentaje Inicial, separando por Algoritmo.
    También añade una línea horizontal discontinua (sin leyenda) para mostrar la media de Accuracy por algoritmo.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Porcentaje Inicial', 'Algoritmo' y 'Accuracy'.
        filename (str | None): Ruta para guardar el gráfico.
        title (str): Título del gráfico.
    """
    if not {"Porcentaje Inicial", "Algoritmo", "Accuracy"}.issubset(df.columns):
        raise ValueError("Faltan columnas necesarias en el DataFrame.")

    plt.figure(figsize=(10, 6))

    # Asegurar que Porcentaje Inicial es numérico
    df = df.copy()
    df["Porcentaje Inicial"] = pd.to_numeric(df["Porcentaje Inicial"], errors="coerce")
    df = df.dropna(subset=["Porcentaje Inicial"])

    # Obtener lista de algoritmos y paleta de colores
    algoritmos = sorted(df["Algoritmo"].unique())
    palette = sns.color_palette(n_colors=len(algoritmos))
    color_map = dict(zip(algoritmos, palette))

    # Lineplot principal con paleta personalizada
    sns.lineplot(
        data=df,
        x="Porcentaje Inicial",
        y="Accuracy",
        hue="Algoritmo",
        palette=color_map,
        marker="o",
        errorbar="sd"
    )

    # Líneas horizontales con media de Accuracy por algoritmo (sin aparecer en leyenda)
    for algoritmo in algoritmos:
        media_acc = df[df["Algoritmo"] == algoritmo]["Accuracy"].mean()
        plt.axhline(
            y=media_acc,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            color=color_map[algoritmo],
            zorder=1
        )

    plt.title(title, fontsize=13)
    plt.xlabel("Porcentaje Inicial", fontsize=11.5)
    plt.ylabel("Accuracy", fontsize=11.5)
    plt.xticks([0.1, 0.25, 0.5, 0.75], fontsize=11.5, fontweight='bold')
    plt.yticks(fontsize=11.5, fontweight='bold')
    plt.grid(True)
    legend = plt.legend(
        title="Algoritmo",
        title_fontsize=11,
        fontsize=10,
        loc='best'
    )
    legend.get_title().set_fontweight('bold')
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        print(f"Lineplot Accuracy guardado en: {filename}")
    plt.close()



def plot_porcentajes_por_algoritmo(
    df: pd.DataFrame,
    tipo: str,
    columnas_clase: list | None = None,
    filename: str | None = None,
    modo: PrintMode = PrintMode.JUNTOS
):
    """
    Genera un gráfico de barras comparando porcentajes agrupados por algoritmo.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        tipo (str): Tipo de gráfico a generar. Puede ser "inicial_final" o "clases".
        columnas_clase (list | None): Columnas de clases (solo para tipo="clases"). Si es None, se usa las columnas de clases por defecto.
        filename (str | None): Ruta del archivo a guardar. Si es None, no se guarda.
        modo (PrintMode): Modo de visualización de los gráficos. Puede ser LIBRES, NO_LIBRES, AMBOS o JUNTOS (por defecto).
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
    df_libres = df[df["Algoritmo"].str.contains(r"(?:libre|-f)$", case=False, regex=True)]
    df_no_libres = df[~df["Algoritmo"].str.contains(r"(?:libre|-f)$", case=False, regex=True)]

    if modo == PrintMode.LIBRES or modo == PrintMode.NO_LIBRES:
        subset_df = df_libres if modo == PrintMode.LIBRES else df_no_libres

        if subset_df.empty:
            print(f"No hay datos para el modo '{modo.value}'.")
            return

        df_melt, hue_col, titulo = preparar(subset_df)
        orden_algoritmos = [alg for alg in ORDEN_ALGORITMOS if alg in df_melt["Algoritmo"].unique()]

        plot_barplot(
            df=df_melt,
            x="Algoritmo",
            y="Porcentaje",
            hue=hue_col,
            order=orden_algoritmos,
            title=f"{titulo} - Algoritmos {modo.value.replace('_', ' ').title()}",
            xlabel="Algoritmo",
            ylabel="Porcentaje (%)",
            filename=filename
        )

    elif modo == PrintMode.AMBOS:
        for subset_df, title in [
            (df_libres, "Algoritmos Libres"),
            (df_no_libres, "Algoritmos No Libres")
        ]:
            if subset_df.empty:
                continue

            df_melt, hue_col, titulo = preparar(subset_df)
            orden_algoritmos = [alg for alg in ORDEN_ALGORITMOS if alg in df_melt["Algoritmo"].unique()]

            plot_barplot(
                df=df_melt,
                x="Algoritmo",
                y="Porcentaje",
                hue=hue_col,
                order=orden_algoritmos,
                title=f"{titulo} - {title}",
                xlabel="Algoritmo",
                ylabel="Porcentaje (%)",
                filename=None
            )

    elif modo == PrintMode.JUNTOS:
        df_melt, hue_col, titulo = preparar(df)
        orden_algoritmos = [alg for alg in ORDEN_ALGORITMOS if alg in df_melt["Algoritmo"].unique()]

        plot_barplot(
            df=df_melt,
            x="Algoritmo",
            y="Porcentaje",
            hue=hue_col,
            order=orden_algoritmos,
            title=f"{titulo} - Todos los algoritmos juntos",
            xlabel="Algoritmo",
            ylabel="Porcentaje (%)",
            filename=filename
        )


def plot_porcentajes_por_porcentaje_inicial(
    df: pd.DataFrame, 
    filename: str | None = None, 
    modo: PrintMode = PrintMode.AMBOS
):
    """
    Genera un gráfico de barras comparando Porcentaje Inicial vs Final,
    agrupado por Porcentaje Inicial. Puede separarse por tipo de algoritmo o mostrarse todo junto.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        filename (str | None): Nombre donde se guardará el gráfico. Si es None, no se guarda.
        modo (PrintMode): Modo de visualización de los gráficos. Puede ser LIBRES, NO_LIBRES, AMBOS (por defecto) o JUNTOS.
    """
    required_columns = {"Porcentaje Inicial", "Porcentaje Final", "Algoritmo"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Faltan columnas necesarias: {', '.join(missing)}")

    df["Porcentaje Inicial"] = df["Porcentaje Inicial"].astype(str)

    df_libres = df[df["Algoritmo"].str.contains(r"(?:libre|-f)$", case=False, regex=True)]
    df_no_libres = df[~df["Algoritmo"].str.contains(r"(?:libre|-f)$", case=False, regex=True)]

    if modo == PrintMode.LIBRES or modo == PrintMode.NO_LIBRES:
        subset_df = df_libres if modo == PrintMode.LIBRES else df_no_libres

        if subset_df.empty:
            print(f"No hay datos para el modo '{modo.value}'.")
            return

        df_melt = subset_df.melt(
            id_vars=["Porcentaje Inicial"],
            value_vars=["Porcentaje Inicial", "Porcentaje Final"],
            var_name="Tipo", value_name="Porcentaje"
        )
        orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: float(x))

        plot_barplot(
            df=df_melt,
            x="Porcentaje Inicial",
            y="Porcentaje",
            hue="Tipo",
            order=orden_x,
            title=f"Porcentaje Inicial vs Final - Algoritmos {modo.value.replace('_', ' ').title()}",
            xlabel="Porcentaje Inicial",
            ylabel="Porcentaje Final",
            filename=filename
        )

    elif modo == PrintMode.AMBOS:
        for subset_df, title in [
            (df_libres, "Algoritmos Libres"),
            (df_no_libres, "Algoritmos No Libres")
        ]:
            if subset_df.empty:
                continue

            df_melt = subset_df.melt(
                id_vars=["Porcentaje Inicial"],
                value_vars=["Porcentaje Inicial", "Porcentaje Final"],
                var_name="Tipo", value_name="Porcentaje"
            )
            orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: float(x))

            plot_barplot(
                df=df_melt,
                x="Porcentaje Inicial",
                y="Porcentaje",
                hue="Tipo",
                order=orden_x,
                title=f"Porcentaje Inicial vs Final - {title}",
                xlabel="Porcentaje Inicial",
                ylabel="Porcentaje Final",
                filename=None
            )

    elif modo == PrintMode.JUNTOS:
        df_melt = df.melt(
            id_vars=["Porcentaje Inicial"],
            value_vars=["Porcentaje Inicial", "Porcentaje Final"],
            var_name="Tipo", value_name="Porcentaje"
        )
        orden_x = sorted(df_melt["Porcentaje Inicial"].unique(), key=lambda x: float(x))

        plot_barplot(
            df=df_melt,
            x="Porcentaje Inicial",
            y="Porcentaje",
            hue="Tipo",
            order=orden_x,
            title="Porcentaje Inicial vs Final - Todos los algoritmos juntos",
            xlabel="Porcentaje Inicial",
            ylabel="Porcentaje Final",
            filename=filename
        )


def guardar_dataframe(
    df_concatenado: pd.DataFrame,
    path_csvs: str
):
    """
    Guarda el DataFrame en un archivo CSV y genera archivos de resumen.
    Args:
        df_concatenado: DataFrame a guardar.
        path_csvs: Ruta donde se guardarán los archivos CSV.
    """
    df_concatenado.to_csv(f'{path_csvs}/concatenado.csv', index=True)
    
    # Convertir Duracion a segundos para incluirla en el promedio
    # Asegurarse de que 'Duracion' sea de tipo timedelta
    df_modified = df_concatenado.copy()
    if not pd.api.types.is_timedelta64_dtype(df_modified["Duracion"]):
        df_modified["Duracion"] = pd.to_timedelta(df_modified["Duracion"], errors='coerce')

    # Convertir Duracion a segundos para incluirla en el promedio
    df_modified["Duracion_segundos"] = df_modified["Duracion"].dt.total_seconds()

    # Agrupar y calcular media incluyendo la duración
    agrupado_media = df_modified.groupby(["Algoritmo", "Porcentaje Inicial"], as_index=False).mean(numeric_only=True)

    # Convertir duración promedio de nuevo a formato timedelta
    agrupado_media["Duracion"] = pd.to_timedelta(agrupado_media["Duracion_segundos"], unit='s')
    agrupado_media.drop(columns=["Duracion_segundos"], inplace=True)

    agrupado_media.to_csv(f"{path_csvs}/media.csv", index=False)

    agrupado_mejor = df_concatenado.sort_values(by="Accuracy", ascending=False).groupby(["Algoritmo", "Porcentaje Inicial"], as_index=False).first()
    agrupado_mejor.to_csv(f"{path_csvs}/mejor.csv", index=False)
    
    return agrupado_media


def plot_metric_vs_algorithms_por_pi(
    df: pd.DataFrame,
    metric: str,
    carpeta_salida: str | None = None,
    titulo: str = "Comparación por Algoritmo para cada Porcentaje Inicial"
):
    """
    Genera un barplot para cada valor de 'Porcentaje Inicial', comparando los algoritmos según una métrica.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'Porcentaje Inicial', 'Algoritmo' y la métrica.
        metric (str): Métrica a mostrar (ej: 'Accuracy').
        carpeta_salida (str | None): Carpeta donde guardar los gráficos. Si es None, no se guardan.
        titulo (str): Título base para los gráficos.
    """
    if not {"Porcentaje Inicial", "Algoritmo", metric}.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas necesarias: 'Porcentaje Inicial', 'Algoritmo', '{metric}'.")

    valores_pi = sorted(df["Porcentaje Inicial"].unique())

    for pi in valores_pi:
        df_filtrado = df[df["Porcentaje Inicial"] == pi]
        orden_algoritmos = [alg for alg in ORDEN_ALGORITMOS if alg in df_filtrado["Algoritmo"].unique()]
        
        filename = None
        if carpeta_salida:
            Path(carpeta_salida).mkdir(parents=True, exist_ok=True)
            filename = f"{carpeta_salida}/BARPLOT-{metric}-algoritmos-PI-{pi}.png"

        plot_barplot(
            df=df_filtrado,
            x="Algoritmo",
            y=metric,
            order=orden_algoritmos,
            title=f"{titulo} (PI={pi})",
            xlabel="Algoritmo",
            ylabel=metric,
            filename=filename,
            mostrar_valores=True
        )
        print("Barplot por PI guardado en: " + filename)


def generate_plots_from_csvs(
    archivos_csv=[
        "results/csvs/resultados_2025-02-23_17-06_task_-1.csv"
    ],
    carpeta_img: str | None = None,
    modelo_name: str | None = None,
    carpeta_csv: str | None = None,
    modo: PrintMode = PrintMode.AMBOS
):
    """
    Genera automáticamente una serie de gráficos a partir de uno o varios archivos CSV con resultados experimentales.

    Esta función concatena los CSV proporcionados, convierte columnas clave a formato numérico, guarda un CSV
    combinado y crea representaciones gráficas (boxplots, barplots, scatter y lineplots) para analizar métricas
    como Accuracy y distribución de clases en función del porcentaje inicial y el algoritmo utilizado.

    Args:
        archivos_csv (list[str]): Lista de rutas a archivos CSV que contienen los resultados experimentales. 
            Por defecto incluye un único archivo de prueba.
        carpeta_img (str | None): Carpeta donde se guardarán los gráficos generados. Si es None, los gráficos no se guardan.
        modelo_name (str | None): Nombre del modelo (usado como prefijo en los nombres de los archivos de imagen).
        carpeta_csv (str | None): Carpeta donde se guardará el CSV combinado y los archivos auxiliares. Si es None, se usa la ruta del primer CSV.
        modo (PrintMode): Modo de visualización de los gráficos. Puede ser LIBRES, NO_LIBRES, AMBOS (por defecto) o JUNTOS.
    """
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
    
    agrupado_media = guardar_dataframe(df_concatenado, path_csvs)

    df = pd.read_csv(f'{path_csvs}/concatenado.csv')

    if modelo_name is None:
        modelo_name = ""

    if carpeta_img is not None:
        filename1 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-porcentaje.png'
        filename2 = f'{carpeta_img}/{modelo_name}-BOXPLOT-accuracy-algoritmo.png'
        filename3 = f'{carpeta_img}/{modelo_name}-BARPLOT-balance-de-clases-por-algoritmo.png'
        filename4 = f'{carpeta_img}/{modelo_name}-BARPLOT-porcentaje-inical-vs-final-por-algoritmo.png'
        filename5 = f'{carpeta_img}/{modelo_name}-BARPLOT-porcentaje-inicial-vs-final-por-pi.png'
        filename6 = filename5.replace("BARPLOT", "LINEPLOT-ACCURACY")
    else:
        filename1 = filename2 = filename3 = filename4 = filename5 = filename6 = None


    # ====== Boxplot 1: Accuracy vs Porcentaje Inicial ======
    plot_boxplot(
        df=df,
        metric="Accuracy",
        eje_x="Porcentaje Inicial",
        hue=None,
        title="Comparación de Accuracy según Porcentaje Inicial y Algoritmo",
        filename=filename1
    )

    # ====== Boxplot 2: Accuracy vs Algoritmo (con hue si está 'Origen') ======
    hue_col = "Origen" if "Origen" in df.columns else None
    plot_boxplot(
        df=df,
        metric="Accuracy",
        eje_x="Algoritmo",
        hue=hue_col,
        title="Comparación de Accuracy según Algoritmo y Origen",
        filename=filename2
    )

    if filename1 and filename2:
        print(f"Los Boxplot se han guardado en {filename1} y {filename2}.")

    plot_metric_vs_algorithms_por_pi(agrupado_media, metric="Accuracy", carpeta_salida=carpeta_img)

    columnas_clase = [col for col in df.columns if col.startswith("Porcentaje ") and col not in ["Porcentaje Inicial", "Porcentaje Final"]]
    
    if "Porcentaje Final" in df.columns and len(columnas_clase) > 1:
        plot_porcentajes_por_algoritmo(df, tipo="clases", filename=filename3, columnas_clase=columnas_clase)
        plot_porcentajes_por_algoritmo(agrupado_media, tipo="inicial_final", filename=filename4, modo=modo)
        plot_porcentajes_por_porcentaje_inicial(agrupado_media, filename=filename5, modo=modo)

        
        print("Se han generado los diagramas de barras.")

        plot_scatter_inicial_final(
            df=df,
            filename=filename5.replace("BARPLOT", "SCATTER"),
            xticks_list=[0.1, 0.25, 0.5, 0.75],
            yticks_list=[0.1, 0.25, 0.5, 0.75, 1],
            ylim_range=(0, 1),
        )

        plot_lineplot_accuracy_por_algoritmo(agrupado_media, filename=filename6)
