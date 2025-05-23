import random

from utils.utils import crear_dict_imagenes
from utils.utils import fitness


def generate_neighbor(current_selection, neighbor_percent, adjust_size: bool = False):
    """
    Genera un vecino modificando la selección actual de imágenes.

    Args:
        current_selection (dict): Diccionario con nombres de imágenes como keys y 0/1 como values
        neighbor_percent (float): Porcentaje mínimo de los vecinos que cambia
        adjust_size (bool): Indica si debe de mantener el número de imágenes seleccionadas

    Returns:
        dict: Nueva selección de imágenes
    """
    # Crear una copia del diccionario actual
    neighbor = current_selection.copy()

    # Obtener listas de imágenes seleccionadas y no seleccionadas
    selected_images = [img for img, val in neighbor.items() if val == 1]
    unselected_images = [img for img, val in neighbor.items() if val == 0]

    modifications_number = int(min(0.15 * len(neighbor), len(selected_images) * neighbor_percent))

    # Realizar las modificaciones
    if adjust_size:
        for _ in range(modifications_number):
            if not selected_images or not unselected_images:
                break

            # Elegir aleatoriamente si añadir o quitar una imagen
            if random.random() < 0.5:
                # Seleccionar una imagen no seleccionada al azar
                img_to_add = random.choice(unselected_images)
                # Actualizar el diccionario y las listas
                neighbor[img_to_add] = 1
                selected_images.append(img_to_add)
                unselected_images.remove(img_to_add)
            else:
                # Deseleccionar una imagen seleccionada al azar
                img_to_remove = random.choice(selected_images)
                # Actualizar el diccionario y las listas
                neighbor[img_to_remove] = 0
                unselected_images.append(img_to_remove)
                selected_images.remove(img_to_remove)
    else:
        for _ in range(modifications_number):
            if not selected_images or not unselected_images:
                break

            # Seleccionar una imagen no seleccionada al azar para añadir
            img_to_add = random.choice(unselected_images)
            # Seleccionar una imagen seleccionada al azar para quitar
            img_to_remove = random.choice(selected_images)

            # Realizar el intercambio
            neighbor[img_to_add] = 1
            neighbor[img_to_remove] = 0

            # Actualizar las listas
            selected_images.remove(img_to_remove)
            selected_images.append(img_to_add)
            unselected_images.remove(img_to_add)
            unselected_images.append(img_to_remove)

    return neighbor


def local_search(
    data_dir: str = "data/RPS/train",
    initial_percentage: int = 10,
    max_evaluations: int = 100,
    max_evaluations_without_improvement: int = 20,
    neighbor_size: int = 10,
    metric: str = "accuracy",
    adjust_size: bool = False,
    model_name: str = "resnet"
) -> tuple[dict, float, list, list, int]:
    """
    Implementa un algoritmo de búsqueda local para selección de imágenes.

    Args:
        data_dir: Diccionaro inicial que contiene las imágenes seleccionadas
        initial_percentage: Número que indica el porcentaje inicial de las imágenes seleccionadas
        max_evaluations: Número máximo de evaluaciones
        max_evaluations_without_improvement: Criterio de parada si no hay mejoras
        neighbor_size: Número de cambios para generar cada vecino
        metric: Métrica a optimizar ("accuracy" o "f1")
        adjust_size: Indica si el porcentaje de imágenes seleccionadas va a variar
        model_name: Nombré del modelo a usar

    Returns:
        tuple: (best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done)
    """

    # Generar y evaluar solución inicial
    current_solution = crear_dict_imagenes(data_dir, initial_percentage)
    current_fitness_dict = fitness(dict_selection=current_solution, model_name=model_name, evaluations=0)
    current_fitness = current_fitness_dict[metric.title()]
    evaluations_done = 1

    best_fitness = current_fitness
    best_solution = current_solution
    fitness_history = [current_fitness_dict]
    best_fitness_history = [best_fitness]

    evaluations_without_improvement = 0

    while evaluations_done < max_evaluations:
        # Generar y evaluar vecino
        neighbor = generate_neighbor(current_solution, neighbor_size, adjust_size)
        neighbor_fitness_dict = fitness(dict_selection=neighbor, model_name=model_name, evaluations=evaluations_done)
        neighbor_fitness = neighbor_fitness_dict[metric.title()]
        evaluations_done += 1

        # Criterio de aceptación
        if neighbor_fitness >= current_fitness:
            current_solution = neighbor.copy()
            current_fitness = neighbor_fitness
            current_fitness_dict = neighbor_fitness_dict

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution
                evaluations_without_improvement = 0
                print(f"Nueva mejor solución encontrada en evaluación {evaluations_done}. Fitness: {best_fitness:.4f}")
            else:
                evaluations_without_improvement += 1
        else:
            evaluations_without_improvement += 1

        fitness_history.append(current_fitness_dict)
        best_fitness_history.append(best_fitness)

        if evaluations_without_improvement >= max_evaluations_without_improvement:
            print(f"Búsqueda terminada por estancamiento después de {evaluations_done} evaluaciones")
            break

    return best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done
