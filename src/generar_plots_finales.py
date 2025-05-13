import argparse
import os

from utils.utils_plot import generate_plots_from_csvs

def recolectar_csvs_de_carpetas(base_dir, carpetas):
    """
    Devuelve una lista con todos los archivos CSV dentro de las carpetas especificadas.
    """
    csvs = []
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(base_dir, carpeta)
        if not os.path.isdir(ruta_carpeta):
            print(f"[AVISO] Carpeta no encontrada: {ruta_carpeta}")
            continue
        archivos = [
            os.path.join(ruta_carpeta, f)
            for f in os.listdir(ruta_carpeta)
            if f.endswith(".csv") and f.startswith("task_")
        ]
        if archivos:
            csvs.extend(archivos)
        else:
            print(f"[AVISO] No se encontraron archivos CSV válidos en {ruta_carpeta}")
    return csvs


def main(fecha_actual, modelo, carpetas_elegidas):
    base_csv_path = "results/csvs/finales_2"
    carpeta_salida = f"img/finales_/{fecha_actual}"
    os.makedirs(carpeta_salida, exist_ok=True)

    todos_los_csvs = recolectar_csvs_de_carpetas(base_csv_path, carpetas_elegidas)

    if not todos_los_csvs:
        print("[ERROR] No se encontraron archivos CSV para las carpetas seleccionadas.")
        return

    print(f"[INFO] Generando gráficos combinados para: {', '.join(carpetas_elegidas)}")
    generate_plots_from_csvs(todos_los_csvs, carpeta_img=carpeta_salida, modelo_name=modelo or "", carpeta_csv=base_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar gráficos combinados desde múltiples carpetas con CSVs.")
    parser.add_argument("--FECHA_ACTUAL", type=str, required=True, help="Fecha actual de ejecución")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Carpetas a combinar (ej: aleatorio bl genetico2)")

    args = parser.parse_args()
    main(args.FECHA_ACTUAL, args.MODELO, args.CARPETAS)
