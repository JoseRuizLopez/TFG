import datetime
import random
from typing import Literal

import torch
import numpy as np

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
    algoritmo: Literal["aleatorio", "busqueda local", "genetico", "memetico"] = "memetico",
    metric: Literal["accuracy", "f1"] = "accuracy",
    model_name: Literal["resnet", "mobilnet"] = "resnet"
):
    set_seed(24012000)

    dataset = "data/dataset/train"

    with open("results/evaluations_logs.txt", "a") as file:
        file.write(f"\n\n---------------------------------------"
                   f"-----------------{algoritmo.upper()}-------"
                   f"---------------------------------------\n\n")
        file.flush()  # Forzar la escritura inmediata al disco

    start = datetime.datetime.now()
    print(f"\n\n--------------------------------------"
          f"----------------{algoritmo.upper()}-------"
          f"------------------------------------------")
    print("Start time: " + str(start))

    if algoritmo == "aleatorio":
        best_selection, best_fitness, fitness_history = random_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "busqueda local":
        best_selection, best_fitness, fitness_history = local_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            neighbor_size=10,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico":
        best_selection, best_fitness, fitness_history = genetic_algorithm(
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
        best_selection, best_fitness, fitness_history = memetic_algorithm(
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
    else:
        best_fitness = 0.0
        best_selection = {}
        fitness_history = []

    end = datetime.datetime.now()
    print("End time: " + str(end))
    print("Duration: " + str(end - start))
    print(f"\n\nMejor {metric} al acabar el algoritmo: {best_fitness:.4f}")

    if best_fitness != 0.0:
        print("\n\nFitness check:\n")
        # Crear y guardar la gráfica
        plot_fitness_evolution(fitness_history, algoritmo, metric)

        final_fitness = fitness(best_selection, metric, model_name=model_name)
        print(f"\n\nMejor {metric} encontrado: {final_fitness:.4f}")
    else:
        print("No se ha seleccionado ningún algoritmo.")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.is_available()}")

    # main(10, 10, 10, "aleatorio", "accuracy", "mobilnet")
    main(10, 100, 100, "busqueda local", "accuracy", "mobilnet")
    # main(10, 100, 10, "genetico", "accuracy", "mobilnet")
    # main(10, 100, 10, "memetico", "accuracy", "mobilnet")
