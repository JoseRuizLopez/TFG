import argparse
import datetime
import os

import numpy as np
import torch
import polars as pl

from src.main import main
from utils.classes import ConfiguracionGlobal
from utils.classes import DatasetList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.utils import plot_multiple_fitness_evolution

if __name__ == "__main__":
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    parser.add_argument("--FECHA_ACTUAL", type=str, required=True, help="Fecha actual para esta ejecución")
    parser.add_argument("--MODELO", type=str, required=False, help="Nombre del modelo (opcional)")

    args = parser.parse_args()

    task_id = args.task_id
    date = args.FECHA_ACTUAL
    model_name = args.MODELO

    print(f"Task ID recibido: {task_id}")

    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10, 20, 50, 100]
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100

    metric: MetricList = MetricList.ACCURACY
    dataset_choosen: DatasetList = DatasetList.PAINTING
    resultados = []
    labels = [str(porcentaje) + '%' for porcentaje in porcentajes]

    now = datetime.datetime.now()
    if os.getenv("SERVER") is not None:
        now = now + datetime.timedelta(hours=1)

    date = now.strftime("%Y-%m-%d_%H-%M")

    # Crear una instancia de ConfiguracionGlobal
    config = ConfiguracionGlobal(date=date, task_id=str(task_id), dataset=dataset_choosen.value)
    carpeta_img = f"img/{date}" + (f"/task_{task_id}" if task_id != -1 else "")

    for model in ModelList:
        fitness_list = []
        best_fitness_list = []
        for ptg in porcentajes:
            result, fitness_history, best_fitness_history = main(
                initial_percentage=ptg,
                max_evaluations=evaluaciones_maximas if ptg != 100 else 1,
                max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
                algoritmo="aleatorio",
                metric=metric.value,
                model_name=model.value,
            )

            result["Accuracy"] = np.mean([fitness["Accuracy"] for fitness in fitness_history])
            result["Precision"] = np.mean([fitness["Precision"] for fitness in fitness_history])
            result["Recall"] = np.mean([fitness["Recall"] for fitness in fitness_history])
            result["F1-score"] = np.mean([fitness["F1-score"] for fitness in fitness_history])

            resultados.append(
                result | {
                    "Porcentaje Inicial": ptg / 100,
                    "Algoritmo": "aleatorio"
                }
            )
            fitness_list.append([fitness[metric.value.title()] for fitness in fitness_history])
            best_fitness_list.append(best_fitness_history)

        plot_multiple_fitness_evolution(
            data=fitness_list,
            labels=labels,
            metric=metric.value,
            title=f'Evolución de cada evaluación - {metric} - Algoritmo {"aleatorio"} - Modelo {model.value} - '
                  f'Con cada porcentaje',
            filename=f'{carpeta_img}/{model.value}-evaluaciones-{"aleatorio".replace(" ", "_")}-combined-{metric}.png'
        )
        plot_multiple_fitness_evolution(
            data=best_fitness_list,
            labels=labels,
            metric=metric.value,
            title=f'Evolución del best {metric} - Algoritmo {"aleatorio"} - Modelo {model.value} - '
                  f'Con cada porcentaje',
            filename=f'{carpeta_img}/{model.value}-best-{"aleatorio"}-combined-{metric}.png'
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
    df = df.rename({
        "Accuracy": "Accuracy (Avg)",
        "Precision": "Precision (Avg)",
        "Recall": "Recall (Avg)",
        "F1-score": "F1-score (Avg)",
    })

    df.write_csv(f"results/csvs/{date}/task_{task_id}.csv")

    print("Se ha creado el Excels con todos los resultados correctamente.")
