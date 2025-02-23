import random

from utils.utils import crear_dict_imagenes
from utils.utils import fitness


def weighted_crossover(parent1: dict, parent2: dict, fitness1: float, fitness2: float) -> tuple[dict, dict]:
    """
    Realiza el cruce entre dos padres donde:
    - El primer hijo toma un porcentaje de imágenes de cada padre basado en sus fitness
    - El segundo hijo toma las imágenes complementarias

    Args:
        parent1, parent2: Diccionarios de selección de imágenes
        fitness1, fitness2: Valores de fitness de los padres

    Returns:
        tuple[dict, dict]: Los dos hijos generados
    """
    # Verificar y corregir si es necesario
    def fix_selection_count(child: dict, target: int) -> dict:
        """Ajusta el número de imágenes seleccionadas al objetivo"""
        selected = set(img for img, val in child.items() if val == 1)
        unselected = set(img for img, val in child.items() if val == 0)

        while len(selected) > target:
            # Quitar imágenes al azar si hay demasiadas
            img_to_remove = random.choice(list(selected))
            child[img_to_remove] = 0
            selected.remove(img_to_remove)
            unselected.add(img_to_remove)

        while len(selected) < target:
            # Añadir imágenes al azar si faltan
            img_to_add = random.choice(list(unselected))
            child[img_to_add] = 1
            selected.add(img_to_add)
            unselected.remove(img_to_add)

        return child

    # Inicializar los hijos como diccionarios vacíos con todas las imágenes en 0
    child1 = {img: 0 for img in parent1.keys()}
    child2 = {img: 0 for img in parent1.keys()}

    # Obtener todas las imágenes seleccionadas de cada padre
    selected1 = set(img for img, val in parent1.items() if val == 1)
    selected2 = set(img for img, val in parent2.items() if val == 1)

    # Calcular el número total de imágenes que deben ser seleccionadas
    target_selected = len(selected1)

    # Calcular los pesos basados en el fitness
    total_fitness = fitness1 + fitness2
    weight1 = fitness1 / total_fitness

    # Calcular cuántas imágenes tomar de cada padre para el primer hijo
    num_from_parent1 = int(target_selected * weight1)
    num_from_parent2 = target_selected - num_from_parent1

    # Seleccionar imágenes aleatorias del padre 1 para el primer hijo
    selected_from_1 = set(random.sample(list(selected1), num_from_parent1))
    remaining_from_1 = selected1 - selected_from_1

    # Seleccionar imágenes aleatorias del padre 2 para el primer hijo
    selected_from_2 = set(random.sample(list(selected2), num_from_parent2))
    remaining_from_2 = selected2 - selected_from_2

    # Construir el primer hijo
    for img in selected_from_1 | selected_from_2:
        child1[img] = 1

    # Construir el segundo hijo con las imágenes restantes
    for img in remaining_from_1 | remaining_from_2:
        child2[img] = 1

    # Asegurar que ambos hijos tienen el número correcto de imágenes seleccionadas

    child1 = fix_selection_count(child1, target_selected)
    child2 = fix_selection_count(child2, target_selected)

    return child1, child2


def mutation(individual: dict, mutation_rate: float = 0.05) -> dict:
    """
    Aplica mutación a un individuo con una determinada probabilidad.
    Mantiene el mismo número de imágenes seleccionadas.
    """
    if random.random() > mutation_rate:
        return individual

    mutated = individual.copy()
    selected = [img for img, val in mutated.items() if val == 1]
    unselected = [img for img, val in mutated.items() if val == 0]

    # Número de intercambios a realizar
    num_swaps = max(1, int(len(selected) * 0.1))

    for _ in range(num_swaps):
        if selected and unselected:
            img_to_remove = random.choice(selected)
            img_to_add = random.choice(unselected)

            mutated[img_to_remove] = 0
            mutated[img_to_add] = 1

            selected.remove(img_to_remove)
            unselected.remove(img_to_add)
            selected.append(img_to_add)
            unselected.append(img_to_remove)

    return mutated


def tournament_selection(
    population: list[dict],
    fitness_values: list[float],
    tournament_size: int = 4
) -> tuple[int, int]:
    """
    Selecciona 2 individuos mediante torneo y devuelve sus índices.

    Returns:
        tuple[int, int]: Índices de los individuos ganadores en la población.
    """
    # Asegurarse de que el tamaño del torneo es suficiente
    tournament_size = max(tournament_size, 3)

    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_results = [(idx, fitness_values[idx]) for idx in tournament_indices]
    tournament_results.sort(key=lambda x: x[1], reverse=True)

    return tournament_results[0][0], tournament_results[1][0]


def genetic_algorithm2(
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
    Algoritmo genético mejorado que da más peso a mejores padres y selecciona el mejor hijo.
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
            # Seleccionar índices de los padres
            parent1_idx, parent2_idx = tournament_selection(population, fitness_values, tournament_size)

            # Obtener los padres y sus valores de fitness
            parent1 = population[parent1_idx].copy()
            parent2 = population[parent2_idx].copy()
            fitness1 = fitness_values[parent1_idx]
            fitness2 = fitness_values[parent2_idx]

            # Generar hijos usando weighted_crossover
            child1, child2 = weighted_crossover(parent1, parent2, fitness1, fitness2)

            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            # Evaluar ambos hijos
            child1_fitness_dict = None
            if evaluations_done + 1 <= max_evaluations:
                child1_fitness_dict = fitness(
                    dict_selection=child1, model_name=model_name, evaluations=evaluations_done
                )
                evaluations_done += 1
                fitness_history.append(child1_fitness_dict)

            child2_fitness_dict = None
            if evaluations_done + 1 <= max_evaluations:
                child2_fitness_dict = fitness(
                    dict_selection=child2, model_name=model_name, evaluations=evaluations_done
                )
                evaluations_done += 1
                fitness_history.append(child2_fitness_dict)

            # Seleccionar el mejor hijo
            if child1_fitness_dict is not None and child2_fitness_dict is not None:
                if child1_fitness_dict[metric.title()] > child2_fitness_dict[metric.title()]:
                    new_population.append(child1)
                    new_fitness_dicts.append(child1_fitness_dict)
                else:
                    new_population.append(child2)
                    new_fitness_dicts.append(child2_fitness_dict)
            elif child1_fitness_dict is not None:
                new_population.append(child1)
                new_fitness_dicts.append(child1_fitness_dict)

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
