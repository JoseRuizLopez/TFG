import datetime
import os
import torch
import polars as pl
import argparse

from src.main import main
from utils.classes import AlgorithmList
from utils.classes import DatasetList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.utils import plot_boxplot
from utils.classes import ConfiguracionGlobal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    task_id = parser.parse_args().task_id

    print(f"Task ID recibido: {task_id}")

    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10]
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100
    add_100 = True

    metric: MetricList = MetricList.ACCURACY
    modelo: ModelList = ModelList.MOBILENET
    dataset_choosen: DatasetList = DatasetList.PAINTING
    resultados = []

    now = datetime.datetime.now()
    if os.getenv("SERVER") is not None:
        now = now + datetime.timedelta(hours=1)
    date = now.strftime("%Y-%m-%d_%H-%M")

    config = ConfiguracionGlobal(date=date, task_id=str(task_id), dataset=dataset_choosen.value)
    carpeta_img = f"img/{date}" + (f"/task_{task_id}" if task_id != -1 else "")

    if add_100:
        result, fitness_history, best_fitness_history = main(
            initial_percentage=100,
            max_evaluations=evaluaciones_maximas,
            max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
            algoritmo=AlgorithmList.ALEATORIO.value,
            metric=metric.value,
            model_name=modelo.value
        )

        resultados.append(
            result | {
                "Porcentaje Inicial": 1,
                "Algoritmo": '-'
            }
        )

    for alg in AlgorithmList:
        for ptg in porcentajes:
            result, fitness_history, best_fitness_history = main(
                initial_percentage=ptg,
                max_evaluations=evaluaciones_maximas,
                max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
                algoritmo=alg.value,
                metric=metric.value,
                model_name=modelo.value
            )

            resultados.append(
                result | {
                    "Porcentaje Inicial": ptg / 100,
                    "Algoritmo": alg.value
                }
            )

    df = pl.DataFrame(resultados, schema={
        "Algoritmo": pl.Utf8,
        "Porcentaje Inicial": pl.Float32,
        "Duracion": pl.Utf8,
        "Accuracy": pl.Float64,
        "Precision": pl.Float64,
        "Recall": pl.Float64,
        "F1-score": pl.Float64,
        "Evaluaciones Realizadas": pl.Int32,
        "Porcentaje Final": pl.Float32
    })

    try:
        # ====== Boxplot 1: Fijando Algoritmo, variando Porcentaje Inicial ======
        plot_boxplot(
            df=df,
            metric="Accuracy",  # O "Precision"
            eje_x="Porcentaje Inicial",
            hue=None,
            title="Comparación de Accuracy según Porcentaje Inicial y Algoritmo",
            filename=f'{carpeta_img}/{modelo.value}-BOXPLOT-accuracy-porcentaje.png',
        )

        # ====== Boxplot 2: Fijando Porcentaje Inicial, variando Algoritmo ======
        plot_boxplot(
            df=df,
            metric="Accuracy",  # O "Precision"
            eje_x="Algoritmo",
            hue=None,
            title="Comparación de Accuracy según Algoritmo y Porcentaje Inicial",
            filename=f'{carpeta_img}/{modelo.value}-BOXPLOT-accuracy-algoritmo.png',
        )
    except Exception as e:
        print("Ha fallado el bloxplot: " + str(e))

    df.write_csv(f"results/csvs/resultados_{date}_task_{task_id}.csv")
    print("Se han generado los boxplots y guardado los resultados correctamente.")
