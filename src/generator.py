import datetime
import os

import torch
import polars as pl
import argparse

from src.main import main
from utils.classes import AlgorithmList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.utils import plot_multiple_fitness_evolution
from utils.classes import ConfiguracionGlobal


if __name__ == "__main__":
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    task_id = parser.parse_args().task_id

    print(f"Task ID recibido: {task_id}")

    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10, 25, 50, 75]
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100
    add_100 = False

    metric: MetricList = MetricList.ACCURACY
    modelo: ModelList = ModelList.MOBILENET
    resultados = []
    labels = [str(porcentaje) + '%' for porcentaje in porcentajes]

    now = datetime.datetime.now()
    if os.getenv("SERVER") is not None:
        now = now + datetime.timedelta(hours=2)

    date = now.strftime("%Y-%m-%d_%H-%M")

    # Crear una instancia de ConfiguracionGlobal
    config = ConfiguracionGlobal(date=date, task_id=str(task_id))
    carpeta_img = f"img/{date}" + (f"/task_{task_id}" if task_id != -1 else "")

    fitness_history_100 = []
    best_fitness_history_100 = []
    if add_100:
        labels.append("100%")
        result, fitness_history_100, best_fitness_history_100 = main(
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
                "Algoritmo": "aleatorio"
            }
        )

    for alg in AlgorithmList:
        fitness_list = []
        best_fitness_list = []
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
            fitness_list.append([fitness[metric.value.title()] for fitness in fitness_history])
            best_fitness_list.append(best_fitness_history)

        if add_100:
            fitness_list.append([fitness[metric.value.title()] for fitness in fitness_history_100])
            best_fitness_list.append(best_fitness_history_100)

        plot_multiple_fitness_evolution(
            data=fitness_list,
            labels=labels,
            algorithm_name=alg.value,
            metric=metric.value,
            model=modelo.value,
            carpeta=carpeta_img,
            selection="mean"
        )
        plot_multiple_fitness_evolution(
            data=best_fitness_list,
            labels=labels,
            algorithm_name=alg.value,
            metric=metric.value,
            model=modelo.value,
            carpeta=carpeta_img,
            selection="best"
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
        "Porcentaje Final": pl.Float32,
        "Porcentaje Paper": pl.Float32,
        "Porcentaje Rock": pl.Float32,
        "Porcentaje Scissors": pl.Float32
    })

    df.write_csv(f"results/csvs/resultados_{date}_task_{task_id}.csv")

    print("Se ha creado el Excels con todos los resultados correctamente.")
