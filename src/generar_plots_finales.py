import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.classes import PrintMode
from utils.utils_plot import generate_plots_from_csvs, plot_boxplot


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


def comparar_algoritmos_por_modelo(carpetas_algoritmos, carpeta_salida, modelo_filtro=None):
    """
    Genera una gráfica comparativa por cada algoritmo de mutación,
    comparando los resultados entre 'finales' y 'finales_2'
    """
    mutation_algorithm_names = {
        "gen_v1": "genetico",
        "gen_v2-2": "genetico2",
        "gen_v2-libre": "genetico2 (libre)",
        "gen_v3": "genetico3",
        "memetico": "memetico",
        "memetico-libre": "memetico-libre"
    }

    base_csv_paths = {
        "genetico2": "results/csvs/finales",
        "genetico2_2": "results/csvs/finales_2",
    }

    for alias, nombre_real in mutation_algorithm_names.items():
        dfs = []
        for origen, base_path in base_csv_paths.items():
            for carpeta in carpetas_algoritmos:
                if carpeta != alias:
                    continue
                carpeta_path = os.path.join(base_path, carpeta)
                if not os.path.isdir(carpeta_path):
                    continue
                archivos = [f for f in os.listdir(carpeta_path) if f.endswith(".csv")]
                for archivo in archivos:
                    path = os.path.join(carpeta_path, archivo)
                    df = pd.read_csv(path)
                    df["Origen"] = origen
                    if modelo_filtro and "Modelo" in df.columns:
                        df = df[df["Modelo"] == modelo_filtro]
                    dfs.append(df)

        if not dfs:
            continue

        df_comb = pd.concat(dfs, ignore_index=True)

        # Verificar que los algoritmos están correctamente filtrados
        df_comb = df_comb[df_comb["Algoritmo"].str.lower() == nombre_real]

        if df_comb.empty:
            continue

        # Generar gráfico comparativo
        output_file = os.path.join(carpeta_salida, f"comparacion_{alias}.png")
        plot_boxplot(
            df=df_comb,
            metric="Accuracy",
            filename=output_file,
            hue=None,
            title=f"Comparación de Accuracy - {alias}",
            eje_x="Origen"
        )
        print(f"Guardado: {output_file}")


def comparar_dos_versiones(output_path, modelo, carpetas_elegidas):
    carpeta_salida_img = f"img/finales/{output_path}"
    os.makedirs(carpeta_salida_img, exist_ok=True)

    comparar_algoritmos_por_modelo(
        carpetas_algoritmos=carpetas_elegidas,
        carpeta_salida=carpeta_salida_img,
        modelo_filtro=modelo
    )
    

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


def main(input_path, output_path, modelo, carpetas_elegidas, modo, modo_print):
    try:
        print_mode = PrintMode(modo_print.lower())
    except Exception as e:
        raise ValueError("El parámetro 'modo' debe ser 'libres', 'no_libres', 'ambos' o 'juntos'.")
    
    if modo == "comparar":
        comparar_dos_versiones(output_path, modelo, carpetas_elegidas)
    elif modo == "individual":
        graficos_una_version(input_path, output_path, modelo, carpetas_elegidas, print_mode)
    else:
        print(f"[ERROR] Modo '{modo}' no reconocido. Usa 'comparar' o 'individual'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar gráficos de algoritmos con o sin comparación de versiones")
    parser.add_argument("--IN", type=str, required=True, help="Nombre para la carpeta de entrada (ej: finales_2)")
    parser.add_argument("--OUT", type=str, required=True, help="Nombre para la carpeta de salida (ej: fecha)")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Carpetas a combinar (ej: gen_v1 gen_v2 mem)")
    parser.add_argument("--MODO", type=str, required=True, choices=["comparar", "individual"], help="Modo de ejecución")
    parser.add_argument("--MODO_PRINT", type=str, required=True, choices=["libres", "no_libres", "ambos", "juntos"], help="Modo de mostrar los gráficos")

    args = parser.parse_args()
    main(input_path=args.IN, output_path=args.OUT, modelo=args.MODELO, carpetas_elegidas=args.CARPETAS, modo=args.MODO, modo_print=args.MODO_PRINT)
