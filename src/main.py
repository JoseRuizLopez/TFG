import argparse
import datetime
import os
import random
from pathlib import Path

import torch
import numpy as np
import polars as pl

from utils.classes import AlgorithmList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.genetic_algorithm import genetic_algorithm
from utils.genetic_algorithm2 import genetic_algorithm2
from utils.genetic_algorithm3 import genetic_algorithm_with_restart
from utils.local_search import local_search
from utils.memetic_algorithm import memetic_algorithm
from utils.random_search import random_search
from utils.utils import fitness
from utils.utils import plot_fitness_evolution
from utils.classes import ConfiguracionGlobal


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(
    initial_percentage: int = 10,
    max_evaluations: int = 10,
    max_evaluations_without_improvement: int = 10,
    algoritmo: str = "memetico",
    metric: str = "accuracy",
    model_name: str = "resnet"
):
    set_seed(24012000)
    config = ConfiguracionGlobal()
    dataset = "data/dataset/train"

    os.makedirs(f"logs/{config.date}", exist_ok=True)
    with open(f"logs/{config.date}/evaluations_log_{config.task_id}.txt", "a") as file:
        file.write(f"\n\n---------------------------------------"
                   f"{model_name}  {algoritmo.upper()}  {str(initial_percentage)}%-------"
                   f"---------------------------------------\n\n")
        file.flush()  # Forzar la escritura inmediata al disco

    start = datetime.datetime.now()
    print(f"\n\n--------------------------------------"
          f"{model_name}  {algoritmo.upper()}  {str(initial_percentage)}%-------"
          f"------------------------------------------")
    print("Start time: " + str(start))

    best_fitness = 0.0
    best_selection = {}
    fitness_history = []
    best_fitness_history = []
    evaluations_done = 0
    if algoritmo == "aleatorio":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = random_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "busqueda local":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = local_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            neighbor_size=10,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = genetic_algorithm(
            data_dir=dataset,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "memetico":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = memetic_algorithm(
            data_dir=dataset,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            local_search_probability=0.2,
            local_search_evaluations=10,
            local_search_neighbor_size=5,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico2":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = genetic_algorithm2(
            data_dir=dataset,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=4,
            mutation_rate=0.05,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico3":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = (
            genetic_algorithm_with_restart(
                data_dir=dataset,
                population_size=5,
                initial_percentage=initial_percentage,
                max_evaluations=max_evaluations,
                max_evaluations_without_improvement=max_evaluations_without_improvement,
                tournament_size=4,
                mutation_rate=0.05,
                metric=metric,
                model_name=model_name
            ))

    end = datetime.datetime.now()
    duration = end - start
    print("End time: " + str(end))
    print("Duration: " + str(duration))
    print(f"\n\nMejor {metric} al acabar el algoritmo: {best_fitness:.4f}")

    if best_fitness != 0.0:
        print("\n\nFitness check:\n")
        os.makedirs("img/" + config.date, exist_ok=True)
        carpeta = f"img/{config.date}" + (f"/task_{config.task_id}" if config.task_id != -1 else "")
        os.makedirs(carpeta, exist_ok=True)
        # Crear y guardar la gráfica
        plot_fitness_evolution(
            fitness_history=best_fitness_history if max_evaluations != 1 else best_fitness_history * 50,
            initial_percentage=initial_percentage,
            algorithm_name=algoritmo,
            metric=metric,
            model=model_name,
            carpeta=carpeta
        )
        if algoritmo == "genetico3":
            plot_fitness_evolution(
                fitness_history=[fit[metric.title()] for fit in fitness_history] if max_evaluations != 1
                else [fit[metric.title()] for fit in fitness_history] * 50,
                initial_percentage=initial_percentage,
                algorithm_name=algoritmo,
                metric=metric,
                model=model_name,
                carpeta=carpeta
            )

        final_fitness = fitness(dict_selection=best_selection, model_name=model_name)
        print(f"\n\nMejor {metric} encontrado: {final_fitness[metric.title()]:.4f}")

        num_images = sum(1 for _ in Path(dataset).rglob('*') if _.is_file())
        images_selected = {key: value for key, value in best_selection.items() if value == 1}
        resultado = final_fitness | {
            "Duracion": str(duration),
            "Evaluaciones Realizadas": evaluations_done,
            "Porcentaje Final": len(images_selected) / num_images,
            "Porcentaje Paper": len(
                [key for key, value in images_selected.items() if 'paper' in key]
            ) / len(images_selected),
            "Porcentaje Rock": len(
                [key for key, value in images_selected.items() if 'rock' in key]
            ) / len(images_selected),
            "Porcentaje Scissors": len(
                [key for key, value in images_selected.items() if 'scissors' in key]
            ) / len(images_selected),
        }

        return resultado, fitness_history, best_fitness_history

    else:
        raise ValueError("No se ha seleccionado ningún algoritmo.")


if __name__ == "__main__":
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    task_id = parser.parse_args().task_id

    print(f"Task ID recibido: {task_id}")

    print(f"GPU: {torch.cuda.is_available()}")
    porcentaje_inicial = 10
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100

    now = datetime.datetime.now()
    if os.getenv("SERVER") is not None:
        now = now + datetime.timedelta(hours=2)

    date = now.strftime("%Y-%m-%d_%H-%M")
    config = ConfiguracionGlobal(date=date, task_id=str(task_id))

    algoritmo: AlgorithmList = AlgorithmList.GENETICO3
    metric: MetricList = MetricList.ACCURACY
    modelo: ModelList = ModelList.MOBILENET

    result, fitness_history, best_fitness_history = main(
        porcentaje_inicial,
        evaluaciones_maximas,
        evaluaciones_maximas_sin_mejora,
        algoritmo.value,
        metric.value,
        modelo.value
    )

    df = pl.DataFrame([result | {
        "Porcentaje Inicial": porcentaje_inicial / 100,
        "Algoritmo": algoritmo.value
    }], schema={
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