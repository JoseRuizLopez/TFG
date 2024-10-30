import random

from utils.utils import fitness
from src.algorithms.genetic_algorithm import crossover
from src.algorithms.genetic_algorithm import mutation
from src.algorithms.genetic_algorithm import tournament_selection
from src.algorithms.local_search import generate_neighbor
from utils.utils import crear_dict_imagenes


def memetic_algorithm(
    data_dir: str = "data/dataset/train",
    population_size: int = 20,
    initial_percentage: int = 10,
    max_evaluations: int = 50,
    max_evaluations_without_improvement: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    local_search_probability: float = 0.2,
    local_search_evaluations: int = 10,
    local_search_neighbor_size: int = 5,
    metric: str = "accuracy",
    model_name: str = "resnet"
) -> tuple[dict, float, list, list, int]:
    """
    Implementa un algoritmo memético para la selección de imágenes.

    Args:
        data_dir: Directorio con las imágenes
        population_size: Tamaño de la población
        initial_percentage: Porcentaje inicial de imágenes a seleccionar
        max_evaluations: Número máximo de evaluaciones
        max_evaluations_without_improvement: Criterio de parada si no hay mejoras
        tournament_size: Tamaño del torneo para selección
        mutation_rate: Probabilidad de mutación
        local_search_probability: Probabilidad de aplicar búsqueda local a un individuo
        local_search_evaluations: Número de evaluaciones de búsqueda local
        local_search_neighbor_size: Tamaño del vecindario en búsqueda local
        metric: Métrica a optimizar ("accuracy" o "f1")
        model_name: Nombré del modelo a usar

    Returns:
        tuple: (best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done)
    """
    # Contador global de evaluaciones
    evaluations_done = 0

    def local_search_improvement_with_limit(individual, remaining_evaluations):
        nonlocal evaluations_done
        current_solution = individual.copy()
        current_fitness_dict = fitness(
            dict_selection=current_solution, model_name=model_name, evaluations=evaluations_done
        )
        current_fitness = current_fitness_dict[metric.title()]
        evaluations_done += 1
        local_evals = 1

        best_solution = current_solution.copy()
        best_fitness_local = current_fitness
        fitness_history_local = [current_fitness_dict.copy()]
        best_fitness_dict_local = current_fitness_dict.copy()

        # Calculamos el número máximo de evaluaciones permitidas para esta búsqueda local
        max_local_evals = min(local_search_evaluations, remaining_evaluations)

        local_fitness_dicts = [current_fitness_dict]

        while local_evals < max_local_evals and evaluations_done < max_evaluations:
            neighbor = generate_neighbor(current_solution, local_search_neighbor_size)
            neighbor_fitness_dict = fitness(
                dict_selection=neighbor, model_name=model_name, evaluations=evaluations_done
            )
            neighbor_fitness = neighbor_fitness_dict[metric.title()]
            evaluations_done += 1
            local_evals += 1

            local_fitness_dicts.append(neighbor_fitness_dict)

            if neighbor_fitness > current_fitness:
                current_solution = neighbor.copy()
                current_fitness = neighbor_fitness

                if current_fitness > best_fitness_local:
                    best_solution = current_solution.copy()
                    best_fitness_local = current_fitness
                    best_fitness_dict_local = neighbor_fitness_dict.copy()

        return best_solution, best_fitness_local, best_fitness_dict_local, local_fitness_dicts, fitness_history_local

    # Generar y evaluar población inicial
    population = [crear_dict_imagenes(data_dir, initial_percentage)
                  for _ in range(population_size)]
    fitness_dicts = [
        fitness(dict_selection=ind, model_name=model_name, evaluations=iteration)
        for ind, iteration in zip(population, range(population_size))
    ]
    fitness_values = [f_dict[metric.title()] for f_dict in fitness_dicts]
    evaluations_done = population_size

    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    best_fitness_dict = fitness_dicts[best_fitness_idx].copy()

    fitness_history = fitness_dicts.copy()
    best_fitness_history = [best_fitness]

    evaluations_without_improvement = 0

    while evaluations_done < max_evaluations:
        new_population = []
        new_fitness_dicts = []

        # Elitismo
        new_population.append(best_individual.copy())
        new_fitness_dicts.append(best_fitness_dict.copy())

        while len(new_population) < population_size and evaluations_done < max_evaluations:
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            for child in [child1, child2]:
                if len(new_population) < population_size and evaluations_done < max_evaluations:
                    if random.random() < local_search_probability:
                        (
                            improved_child,
                            child_fitness,
                            child_fitness_dict,
                            local_fitness_dicts,
                            local_fitness_history
                        ) = local_search_improvement_with_limit(
                            child,
                            max_evaluations - evaluations_done
                        )
                        new_population.append(improved_child)
                        new_fitness_dicts.append(child_fitness_dict)
                        fitness_history.extend(local_fitness_history)
                    else:
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
            best_fitness_dict = fitness_dicts[current_best_idx].copy()
            evaluations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en evaluación {evaluations_done}. Fitness: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1

        # fitness_history.extend(fitness_dicts)
        best_fitness_history.append(best_fitness)

        print(f"Evaluaciones realizadas: {evaluations_done}/{max_evaluations}")
        print(f"Mejor fitness actual: {best_fitness:.4f}")
        print(f"Evaluaciones sin mejora: {evaluations_without_improvement}")

        if evaluations_without_improvement >= max_evaluations_without_improvement:
            print(f"Búsqueda terminada por estancamiento después de {evaluations_done} evaluaciones")
            break

    return best_individual, best_fitness, fitness_history, best_fitness_history, evaluations_done
