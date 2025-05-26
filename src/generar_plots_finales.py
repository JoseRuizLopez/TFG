import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classes import PrintMode
from utils.utils_plot import generate_plots_from_csvs


def recolectar_csvs_de_carpetas(base_dir, carpetas, origen):
    csvs = []
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(base_dir, carpeta)
        if not os.path.isdir(ruta_carpeta):
            print(f"[AVISO] Carpeta no encontrada: {ruta_carpeta}")
            continue
        archivos = [
            (os.path.join(ruta_carpeta, f), origen)
            for f in os.listdir(ruta_carpeta)
            if f.endswith(".csv") and f.startswith("task_")
        ]
        if archivos:
            csvs.extend(archivos)
        else:
            print(f"[AVISO] No se encontraron archivos CSV válidos en {ruta_carpeta}")
    return csvs


def graficos_una_version(input_path, output_path, modelo, carpetas_elegidas, print_mode):
    carpeta_salida_img = f"img/finales/{output_path}"
    base_csv_path = f"results/csvs/{input_path}"
    os.makedirs(carpeta_salida_img, exist_ok=True)

    csvs_con_origen = recolectar_csvs_de_carpetas(base_csv_path, carpetas_elegidas, origen="version_2")
    todos_los_csvs = [ruta for ruta, _ in csvs_con_origen]

    if not todos_los_csvs:
        print("[ERROR] No se encontraron archivos CSV para las carpetas seleccionadas.")
        return

    print(f"Generando gráficos combinados para: {', '.join(carpetas_elegidas)}")
    generate_plots_from_csvs(
        archivos_csv=todos_los_csvs,
        carpeta_img=carpeta_salida_img,
        modelo_name=modelo or "",
        carpeta_csv=base_csv_path,
        modo=print_mode
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar gráficos de una sola versión de resultados")
    parser.add_argument("--IN", type=str, required=True, help="Nombre para la carpeta de entrada (ej: finales_2)")
    parser.add_argument("--OUT", type=str, required=True, help="Nombre para la carpeta de salida (ej: fecha)")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Carpetas a combinar (ej: gen_v1 gen_v2 mem)")
    parser.add_argument("--MODO_PRINT", type=str, required=True, choices=["libres", "no_libres", "ambos", "juntos"], help="Modo de mostrar los gráficos")

    args = parser.parse_args()

    try:
        print_mode = PrintMode(args.MODO_PRINT.lower())
    except Exception:
        raise ValueError("El parámetro 'MODO_PRINT' debe ser 'libres', 'no_libres', 'ambos' o 'juntos'.")

    graficos_una_version(
        input_path=args.IN,
        output_path=args.OUT,
        modelo=args.MODELO,
        carpetas_elegidas=args.CARPETAS,
        print_mode=print_mode
    )
