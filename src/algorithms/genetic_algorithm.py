import random

from utils.utils import crear_dict_imagenes
from utils.utils import fitness
from utils.utils import mutation


def crossover(parent1: dict, parent2: dict) -> tuple[dict, dict]:
    """
    Realiza el cruce entre dos padres para generar dos hijos.
    Mantiene el mismo número de imágenes seleccionadas en los hijos.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Obtener las listas de imágenes seleccionadas de cada padre
    selected1 = set(img for img, val in parent1.items() if val == 1)
    selected2 = set(img for img, val in parent2.items() if val == 1)

    # Calcular las diferencias entre los conjuntos
    only_in_1 = selected1 - selected2
    only_in_2 = selected2 - selected1

    # Intercambiar aleatoriamente algunas imágenes diferentes
    num_swap = min(len(only_in_1), len(only_in_2)) // 2

    if num_swap > 0:
        swap_from_1 = random.sample(list(only_in_1), num_swap)
        swap_from_2 = random.sample(list(only_in_2), num_swap)

        # Realizar el intercambio en child1
        for img1, img2 in zip(swap_from_1, swap_from_2):
            child1[img1] = 0
            child1[img2] = 1

        # Realizar el intercambio en child2
        for img1, img2 in zip(swap_from_1, swap_from_2):
            child2[img2] = 0
            child2[img1] = 1

    return child1, child2


def tournament_selection(population: list[dict], fitness_values: list[float], tournament_size: int = 3) -> dict:
    """
    Selecciona un individuo mediante torneo.
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx].copy()


def genetic_algorithm(
    data_dir: str = "data/RPS/train",
    population_size: int = 10,
    initial_percentage: int = 10,
    max_evaluations: int = 50,
    max_evaluations_without_improvement: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    metric: str = "accuracy",
    model_name: str = "resnet"
) -> tuple[dict, float, list, list, int]:
    """
    Implementa un algoritmo genético para la selección de imágenes.

    Args:
        data_dir: Directorio con las imágenes
        population_size: Tamaño de la población
        initial_percentage: Porcentaje inicial de imágenes a seleccionar
        max_evaluations: Número máximo de evaluaciones
        max_evaluations_without_improvement: Criterio de parada si no hay mejoras
        tournament_size: Tamaño del torneo para selección
        mutation_rate: Probabilidad de mutación
        metric: Métrica a optimizar ("accuracy" o "f1")
        model_name: Nombré del modelo a usar

    Returns:
        tuple: (best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done)
    """
    # Generar y evaluar población inicial
    population = [crear_dict_imagenes(data_dir, initial_percentage)
                  for _ in range(min(population_size, max_evaluations))]
    fitness_dicts = [
        fitness(dict_selection=ind, model_name=model_name, evaluations=iteration)
        for ind, iteration in zip(population, range(len(population)))
    ]
    fitness_values = [f_dict[metric.title()] for f_dict in fitness_dicts]
    evaluations_done = len(population)

    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    fitness_history = fitness_dicts.copy()
    best_fitness_history = [best_fitness]

    evaluations_without_improvement = 0

    while evaluations_done < max_evaluations:
        new_population = []
        new_fitness_dicts = []

        # Elitismo
        new_population.append(best_individual.copy())
        new_fitness_dicts.append(fitness_dicts[best_fitness_idx])

        # Generar nueva población
        while len(new_population) < population_size and evaluations_done < max_evaluations:
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            for child in [child1, child2]:
                if len(new_population) < population_size and evaluations_done < max_evaluations:
                    child_fitness_dict = fitness(
                        dict_selection=child, model_name=model_name, evaluations=evaluations_done
                    )
                    evaluations_done += 1
                    new_population.append(child)
                    new_fitness_dicts.append(child_fitness_dict)
                    fitness_history.append(child_fitness_dict)

        population = new_population
        fitness_dicts = new_fitness_dicts
        fitness_values = [f_dict[metric.title()] for f_dict in fitness_dicts]

        current_best_idx = fitness_values.index(max(fitness_values))
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            evaluations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en evaluación {evaluations_done}. Fitness: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1

        # fitness_history.extend(fitness_dicts)
        best_fitness_history.append(best_fitness)

        if evaluations_without_improvement >= max_evaluations_without_improvement:
            print(f"Búsqueda terminada por estancamiento después de {evaluations_done} evaluaciones")
            break

    return best_individual, best_fitness, fitness_history, best_fitness_history, evaluations_done
