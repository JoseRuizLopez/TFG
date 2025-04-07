import random

from src.algorithms.genetic_algorithm2 import tournament_selection
from utils.utils import crear_dict_imagenes
from utils.utils import fitness
from utils.utils import mutation


def weighted_crossover(parent1: dict, parent2: dict, fitness1: float, fitness2: float) -> tuple[dict, dict]:
    """
    Realiza el cruce entre dos padres donde:
    - El primer hijo toma un porcentaje de imágenes de cada padre basado en sus fitness
    - El segundo hijo toma las imágenes complementarias
    """

    # El resto de la función weighted_crossover permanece igual
    def fix_selection_count(child: dict, target: int) -> dict:
        selected = set(img for img, val in child.items() if val == 1)
        unselected = set(img for img, val in child.items() if val == 0)

        while len(selected) > target:
            img_to_remove = random.choice(list(selected))
            child[img_to_remove] = 0
            selected.remove(img_to_remove)
            unselected.add(img_to_remove)

        while len(selected) < target:
            img_to_add = random.choice(list(unselected))
            child[img_to_add] = 1
            selected.add(img_to_add)
            unselected.remove(img_to_add)

        return child

    child1 = {img: 0 for img in parent1.keys()}
    child2 = {img: 0 for img in parent1.keys()}

    selected1 = set(img for img, val in parent1.items() if val == 1)
    selected2 = set(img for img, val in parent2.items() if val == 1)

    target_selected = len(selected1)

    total_fitness = fitness1 + fitness2
    weight1 = fitness1 / total_fitness

    num_from_parent1 = int(target_selected * weight1)
    num_from_parent2 = target_selected - num_from_parent1

    selected_from_1 = set(random.sample(list(selected1), num_from_parent1))
    remaining_from_1 = selected1 - selected_from_1

    selected_from_2 = set(random.sample(list(selected2), num_from_parent2))
    remaining_from_2 = selected2 - selected_from_2

    for img in selected_from_1 | selected_from_2:
        child1[img] = 1

    for img in remaining_from_1 | remaining_from_2:
        child2[img] = 1

    child1 = fix_selection_count(child1, target_selected)
    child2 = fix_selection_count(child2, target_selected)

    return child1, child2


def genetic_algorithm_with_restart(
    data_dir: str = "data/RPS/train",
    population_size: int = 10,
    initial_percentage: int = 10,
    max_evaluations: int = 50,
    max_evaluations_without_improvement: int = 10,
    tournament_size: int = 4,
    mutation_rate: float = 0.05,
    metric: str = "accuracy",
    model_name: str = "resnet"
) -> tuple[dict, float, list, list, int]:
    """
    Algoritmo genético mejorado con reinicio de población cuando el segundo mejor no mejora en 2 generaciones.
    """

    def initialize_population(size: int, keep_best: bool = False) -> tuple[list, list, list, list]:
        """
        Inicializa o reinicia la población, opcionalmente manteniendo el mejor individuo.
        """
        print("Inicializando población...")
        new_pop = []
        new_local_fitness_dicts: list[dict] = []

        # Si debemos mantener el mejor individuo, lo añadimos primero
        if keep_best and 'best_individual' in locals():
            new_pop.append(best_individual)
            new_local_fitness_dicts.append(best_fitness_dict)
            remaining_size = min(size - 1, max_evaluations - evaluations_done)
        else:
            remaining_size = min(size, max_evaluations - evaluations_done)

        # Generar el resto de la población aleatoriamente
        random_population = [crear_dict_imagenes(data_dir, initial_percentage)
                             for _ in range(remaining_size)]

        # Evaluar la población nueva
        random_fitness_dicts = [
            fitness(dict_selection=ind, model_name=model_name, evaluations=evaluations_done + i)
            for i, ind in enumerate(random_population)
        ]

        new_pop.extend(random_population)
        new_local_fitness_dicts.extend(random_fitness_dicts)
        new_local_fitness_history = random_fitness_dicts
        new_fitness_values = [f_dict[metric.title()] for f_dict in new_local_fitness_dicts]

        if keep_best and 'population' in locals() and len(new_pop) < size:
            # Calcular cuántos individuos adicionales necesitamos
            slots_remaining = size - len(new_pop)

            # Añadir los mejores de la población anterior (excluyendo el mejor que ya añadimos)
            new_pop.extend([population[idx].copy() for idx in sorted_indices[1:slots_remaining+1]])
            new_local_fitness_dicts.extend([fitness_dicts[idx].copy() for idx in sorted_indices[1:slots_remaining+1]])
            new_fitness_values.extend([fitness_values[idx] for idx in sorted_indices[1:slots_remaining+1]])

        return new_pop, new_local_fitness_dicts, new_fitness_values, new_local_fitness_history

    # Inicialización
    evaluations_done = 0
    generations_second_best_not_improved = 0

    population, fitness_dicts, fitness_values, fitness_history = initialize_population(population_size)
    evaluations_done += population_size

    # Encontrar el mejor y segundo mejor inicial
    sorted_indices = sorted(range(len(fitness_values)),
                            key=lambda k: fitness_values[k],
                            reverse=True)
    best_idx, second_best_idx = sorted_indices[0], sorted_indices[1]

    best_individual = population[best_idx].copy()
    best_fitness = fitness_values[best_idx]
    best_fitness_dict = fitness_dicts[best_idx].copy()
    second_best_fitness = fitness_values[second_best_idx]

    best_fitness_history = [best_fitness]
    evaluations_without_improvement = 0
    iterations = 0

    while evaluations_done < max_evaluations:
        print("Iteration " + str(iterations))
        new_population = []
        new_fitness_dicts = []

        # Elitismo
        new_population.append(best_individual.copy())
        new_fitness_dicts.append(best_fitness_dict.copy())

        # Generar nueva población
        while len(new_population) < population_size and evaluations_done < max_evaluations:
            parent1_idx, parent2_idx = tournament_selection(population, fitness_values, tournament_size)
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()
            fitness1 = fitness_values[parent1_idx]
            fitness2 = fitness_values[parent2_idx]

            child1, child2 = weighted_crossover(parent1, parent2, fitness1, fitness2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            remaining_evals = max_evaluations - evaluations_done
            child1_fitness_dict = None
            child2_fitness_dict = None

            if remaining_evals >= 2:
                child1_fitness_dict = fitness(
                    dict_selection=child1, model_name=model_name, evaluations=evaluations_done
                )
                evaluations_done += 1

                child2_fitness_dict = fitness(
                    dict_selection=child2, model_name=model_name, evaluations=evaluations_done
                )
                evaluations_done += 1

                fitness_history.extend([child1_fitness_dict.copy(), child2_fitness_dict.copy()])

            elif remaining_evals == 1:
                child1_fitness_dict = fitness(
                    dict_selection=child1, model_name=model_name, evaluations=evaluations_done
                )
                evaluations_done += 1
                fitness_history.append(child1_fitness_dict.copy())

            if child1_fitness_dict and child2_fitness_dict:
                if child1_fitness_dict[metric.title()] > child2_fitness_dict[metric.title()]:
                    new_population.append(child1)
                    new_fitness_dicts.append(child1_fitness_dict)
                else:
                    new_population.append(child2)
                    new_fitness_dicts.append(child2_fitness_dict)
            elif child1_fitness_dict:
                new_population.append(child1)
                new_fitness_dicts.append(child1_fitness_dict)

        population = new_population
        fitness_dicts: list[dict] = new_fitness_dicts
        fitness_values = [f_dict[metric.title()] for f_dict in fitness_dicts]

        # Encontrar el mejor y segundo mejor actual
        sorted_indices = sorted(range(len(fitness_values)),
                                key=lambda k: fitness_values[k],
                                reverse=True)
        current_best_idx = sorted_indices[0]
        current_second_best_idx = sorted_indices[1]
        # current_second_best_fitness = fitness_values[current_second_best_idx]

        # Verificar si hay mejora en el mejor individuo
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            best_fitness_dict = fitness_dicts[current_best_idx].copy()
            evaluations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en evaluación {evaluations_done}. Fitness: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1

        # Verificar si el segundo mejor ha mejorado
        if fitness_values[current_second_best_idx] <= second_best_fitness:
            generations_second_best_not_improved += 1
            print(f"El segundo mejor no mejoró. Contador: {generations_second_best_not_improved}/2")
        else:
            generations_second_best_not_improved = 0
            second_best_fitness = fitness_values[current_second_best_idx]

            # Reiniciar población manteniendo solo el mejor
            population, fitness_dicts, fitness_values, local_fitness_history = initialize_population(
                population_size, keep_best=True
            )
            evaluations_done += population_size - 1  # -1 porque ya contamos el mejor

            # Encontrar el mejor y segundo mejor inicial
            sorted_indices = sorted(range(len(fitness_values)),
                                    key=lambda k: fitness_values[k],
                                    reverse=True)
            best_idx, second_best_idx = sorted_indices[0], sorted_indices[1]

            best_individual = population[best_idx].copy()
            best_fitness = fitness_values[best_idx]
            best_fitness_dict = fitness_dicts[best_idx].copy()
            second_best_fitness = fitness_values[second_best_idx]

            best_fitness_history = [best_fitness]
            fitness_history.extend(local_fitness_history)

            generations_second_best_not_improved = 0

        best_fitness_history.append(best_fitness)
        iterations += 1

        if evaluations_without_improvement >= max_evaluations_without_improvement:
            print(f"Búsqueda terminada por estancamiento después de {evaluations_done} evaluaciones")
            break

    return best_individual, best_fitness, fitness_history, best_fitness_history, evaluations_done
