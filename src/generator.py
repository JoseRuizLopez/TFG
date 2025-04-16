import datetime
import os
import random

import torch
import polars as pl
import argparse

from src.main import main
from utils.classes import AlgorithmList
from utils.classes import DatasetList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.classes import ConfiguracionGlobal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    parser.add_argument("--FECHA_ACTUAL", type=str, required=False, help="Fecha actual para esta ejecución")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")

    args = parser.parse_args()

    task_id = args.task_id
    date = args.FECHA_ACTUAL
    model_name = args.MODELO

    if not date:
        date = datetime.datetime.now().strftime("%Y/%m/%d/%H-%M")
    if model_name:
        modelo: ModelList = ModelList(model_name)
    else:
        modelo: ModelList = ModelList.MOBILENET

    print(f"Task ID recibido: {task_id}")

    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10, 25, 50, 75]
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100
    add_100 = True

    metric: MetricList = MetricList.ACCURACY
    dataset_choosen: DatasetList = DatasetList.RPS

    resultados = []

    config = ConfiguracionGlobal(date=date, task_id=str(task_id), dataset=dataset_choosen.value)
    carpeta_img = f"img/{date}" + (f"/task_{task_id}" if task_id != -1 else "")

    if add_100:
        result, fitness_history, best_fitness_history = main(
            initial_percentage=100,
            max_evaluations=1,
            max_evaluations_without_improvement=1,
            algoritmo="aleatorio",
            metric=metric.value,
            model_name=modelo.value
        )

        resultados.append(
            result | {
                "Porcentaje Inicial": 1,
                "Algoritmo": '-'
            }
        )

    if "FREE_BUSQUEDA_LOCAL" in AlgorithmList.__members__:
        random.seed(24012000 + 1)
        random_initial_percentage = random.randint(1, 100)
        result, fitness_history, best_fitness_history = main(
            initial_percentage=random_initial_percentage,
            max_evaluations=evaluaciones_maximas,
            max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
            algoritmo=AlgorithmList.FREE_BUSQUEDA_LOCAL.value,
            metric=metric.value,
            model_name=modelo.value,
            vary_percentage=False
        )

        resultados.append(
            result | {
                "Porcentaje Inicial": random_initial_percentage / 100,
                "Algoritmo": AlgorithmList.FREE_BUSQUEDA_LOCAL.value
            }
        )

    for alg in AlgorithmList:
        if alg.value != "free busqueda local":
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
    schema = {
        "Algoritmo": pl.Utf8,
        "Porcentaje Inicial": pl.Float64,
        "Duracion": pl.Utf8,
        "Accuracy": pl.Float64,
        "Precision": pl.Float64,
        "Recall": pl.Float64,
        "F1-score": pl.Float64,
        "Evaluaciones Realizadas": pl.Int32,
        "Porcentaje Final": pl.Float64,  # len(images_selected) / num_images
    } | {
        "Porcentaje " + name.capitalize(): pl.Float64
        for name in os.listdir(config.dataset + '/train')
        if os.path.isdir(os.path.join(config.dataset + '/train', name))
    }

    df = pl.DataFrame(resultados, schema=schema).with_columns(
        pl.col(pl.Float64).round(4)
    )

    result_csv = f"results/csvs/{date}/task_{task_id}.csv"
    df.write_csv(result_csv)
    print("Se ha generado el CSV con los resultados correctamente.")
