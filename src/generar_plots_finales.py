import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        "memetico": "memetico"
    }

    base_csv_paths = {
        "version_1": "results/csvs/finales",
        "version_2": "results/csvs/finales_2",
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
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_comb, x="Origen", y="Accuracy")
        plt.title(f"Comparación de Accuracy - {alias}")
        plt.ylabel("Accuracy")
        plt.xlabel("Origen")
        plt.grid(True)
        plt.tight_layout()

        output_file = os.path.join(carpeta_salida, f"comparacion_{alias}.png")
        plt.savefig(output_file)
        print(f"[OK] Guardado: {output_file}")
        plt.close()


def comparar_dos_versiones(OUT, modelo, carpetas_elegidas):
    carpeta_salida_img = f"img/finales/{OUT}"
    os.makedirs(carpeta_salida_img, exist_ok=True)

    comparar_algoritmos_por_modelo(
        carpetas_algoritmos=carpetas_elegidas,
        carpeta_salida=carpeta_salida_img,
        modelo_filtro=modelo
    )


def graficos_una_version(OUT, modelo, carpetas_elegidas):
    carpeta_salida_img = f"img/finales/{OUT}"
    base_csv_path = "results/csvs/finales_2"
    os.makedirs(carpeta_salida_img, exist_ok=True)

    csvs_con_origen = recolectar_csvs_de_carpetas(base_csv_path, carpetas_elegidas, origen="version_2")
    todos_los_csvs = [ruta for ruta, _ in csvs_con_origen]

    if not todos_los_csvs:
        print("[ERROR] No se encontraron archivos CSV para las carpetas seleccionadas.")
        return

    print(f"[INFO] Generando gráficos combinados para: {', '.join(carpetas_elegidas)}")
    generate_plots_from_csvs(
        archivos_csv=todos_los_csvs,
        carpeta_img=carpeta_salida_img,
        modelo_name=modelo or "",
        carpeta_csv=base_csv_path
    )


def main(OUT, modelo, carpetas_elegidas, modo):
    if modo == "comparar":
        comparar_dos_versiones(OUT, modelo, carpetas_elegidas)
    elif modo == "individual":
        graficos_una_version(OUT, modelo, carpetas_elegidas)
    else:
        print(f"[ERROR] Modo '{modo}' no reconocido. Usa 'comparar' o 'individual'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar gráficos de algoritmos con o sin comparación de versiones")
    parser.add_argument("--OUT", type=str, required=True, help="Nombre para la carpeta de salida (ej: fecha)")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")
    parser.add_argument("--CARPETAS", nargs='+', required=True, help="Carpetas a combinar (ej: gen_v1 gen_v2 mem)")
    parser.add_argument("--MODO", type=str, required=True, choices=["comparar", "individual"], help="Modo de ejecución")

    args = parser.parse_args()
    main(args.OUT, args.MODELO, args.CARPETAS, args.MODO)
