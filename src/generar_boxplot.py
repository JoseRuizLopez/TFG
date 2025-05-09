import argparse

from src.pruebas import generate_boxplot_from_csvs


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

    archivos_csv = [f"results/csvs/{FECHA_ACTUAL}/task_{str(x)}.csv" for x in range(5)]

    generate_boxplot_from_csvs(archivos_csv, f"img/{FECHA_ACTUAL}", modelo_name=MODELO)
