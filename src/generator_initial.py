import datetime

import numpy as np
import torch
import polars as pl

from src.main import main
from utils.classes import MetricList
from utils.classes import ModelList
from utils.utils import plot_multiple_fitness_evolution

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10, 20, 50, 100]
    evaluaciones_maximas = 5
    evaluaciones_maximas_sin_mejora = 5

    metric: MetricList = MetricList.ACCURACY
    resultados = []
    labels = [str(porcentaje) + '%' for porcentaje in porcentajes] + ["100%"]
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    for model in ModelList:
        fitness_list = []
        best_fitness_list = []
        for ptg in porcentajes:
            result, fitness_history, best_fitness_history = main(
                initial_percentage=ptg,
                max_evaluations=evaluaciones_maximas,
                max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
                algoritmo="aleatorio",
                metric=metric.value,
                model_name=model.value,
                date=date
            )

            result["Accuracy"] = np.mean([fitness["Accuracy"] for fitness in fitness_history])
            result["Precision"] = np.mean([fitness["Precision"] for fitness in fitness_history])
            result["Recall"] = np.mean([fitness["Recall"] for fitness in fitness_history])
            result["F1-score"] = np.mean([fitness["F1-score"] for fitness in fitness_history])

            resultados.append(
                result | {
                    "Porcentaje Inicial": ptg,
                    "Algoritmo": "aleatorio"
                }
            )
            fitness_list.append([fitness[metric.value.title()] for fitness in fitness_history])
            best_fitness_list.append(best_fitness_history)

        plot_multiple_fitness_evolution(
            fitness_list, labels, "aleatorio", metric.value, model.value, "mean"
        )
        plot_multiple_fitness_evolution(
            best_fitness_list, labels, "aleatorio", metric.value, model.value, "best"
        )

    df = pl.DataFrame(resultados, schema={
        "Algoritmo": pl.Utf8,
        "Porcentaje Inicial": pl.Int32,
        "Duracion": pl.Utf8,
        "Accuracy (Avg)": pl.Float64,
        "Precision (Avg)": pl.Float64,
        "Recall (Avg)": pl.Float64,
        "F1-score (Avg)": pl.Float64,
        "Evaluaciones Realizadas": pl.Int32,
        "Porcentaje Final": pl.Float32,
        "Porcentaje Paper": pl.Float32,
        "Porcentaje Rock": pl.Float32,
        "Porcentaje Scissors": pl.Float32
    })

    df.write_csv(f"results/resultados_{date}.csv")

    print("Se ha creado el Excels con todos los resultados correctamente.")
