import argparse
import os
import matplotlib
matplotlib.use('Agg')

from utils.classes import PrintMode
from utils.utils_plot import generate_plots_from_csvs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de generación de boxplots")

    # Define all arguments before parsing
    parser.add_argument("--FECHA_ACTUAL", type=str, required=True, help="Fecha actual para esta ejecución")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")

    # Parse all arguments at once
    args = parser.parse_args()

    # Access the parsed arguments
    FECHA_ACTUAL = args.FECHA_ACTUAL
    MODELO = args.MODELO

    path = f"results/csvs/{FECHA_ACTUAL}"
    path_img = f"img/{FECHA_ACTUAL}"

    archivos_csv = [f"{path}/{f}" for f in os.listdir(path) if f.startswith("task_")]

    os.makedirs(path_img, exist_ok=True)
    generate_plots_from_csvs(archivos_csv, path_img, modelo_name=MODELO, modo=PrintMode.AMBOS)
