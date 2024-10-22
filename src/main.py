import datetime
import os
import random
from pathlib import Path

import torch
import numpy as np

from utils.classes import AlgorithmList
from utils.classes import MetricList
from utils.classes import ModelList
from utils.genetic_algorithm import genetic_algorithm
from utils.local_search import local_search
from utils.memetic_algorithm import memetic_algorithm
from utils.random_search import random_search
from utils.utils import fitness
from utils.utils import plot_fitness_evolution


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
    model_name: str = "resnet",
    date: str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
):
    set_seed(24012000)

    dataset = "data/dataset/train"

    with open(f"logs/evaluations_logs.txt", "a") as file:
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

    end = datetime.datetime.now()
    duration = end - start
    print("End time: " + str(end))
    print("Duration: " + str(duration))
    print(f"\n\nMejor {metric} al acabar el algoritmo: {best_fitness:.4f}")

    if best_fitness != 0.0:
        print("\n\nFitness check:\n")
        if not os.path.exists("img/" + date):
            os.mkdir("img/" + date)
        # Crear y guardar la gráfica
        plot_fitness_evolution(
            fitness_history=best_fitness_history if max_evaluations != 1 else best_fitness_history * 50,
            initial_percentage=initial_percentage,
            algorithm_name=algoritmo,
            metric=metric,
            model=model_name,
            carpeta=date
        )

        final_fitness = fitness(dict_selection=best_selection, model_name=model_name)
        print(f"\n\nMejor {metric} encontrado: {final_fitness[metric.title()]:.4f}")

        num_images = sum(1 for _ in Path(dataset).rglob('*') if _.is_file())
        images_selected = {key: value for key, value in best_selection.items() if value == 1}
        resultado = final_fitness | {
            "Duracion": str(duration),
            "Evaluaciones Realizadas": evaluations_done,
            "Porcentaje Final":  len(images_selected) / num_images * 100,
            "Porcentaje Paper": len(
                [key for key, value in images_selected.items() if 'paper' in key]
            ) / len(images_selected) * 100,
            "Porcentaje Rock": len(
                [key for key, value in images_selected.items() if 'rock' in key]
            ) / len(images_selected) * 100,
            "Porcentaje Scissors": len(
                [key for key, value in images_selected.items() if 'scissors' in key]
            ) / len(images_selected) * 100,
        }

        return resultado, fitness_history, best_fitness_history

    else:
        raise ValueError("No se ha seleccionado ningún algoritmo.")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.is_available()}")
    porcentaje_inicial = 10
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 100

    algoritmo: AlgorithmList = AlgorithmList.ALEATORIO
    metric: MetricList = MetricList.ACCURACY
    modelo: ModelList = ModelList.RESNET

    result, fitness_history, best_fitness_history = main(
        porcentaje_inicial,
        evaluaciones_maximas,
        evaluaciones_maximas_sin_mejora,
        algoritmo.value,
        metric.value,
        modelo.value
    )
