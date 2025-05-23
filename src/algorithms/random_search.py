from utils.utils import fitness
from utils.utils import crear_dict_imagenes


def random_search(
    data_dir: str = "data/RPS/train",
    initial_percentage: int = 10,
    max_evaluations: int = 100,
    max_evaluations_without_improvement: int = 20,
    metric: str = "accuracy",
    model_name: str = "resnet"
) -> tuple[dict, float, list, list, int]:
    """
    Implementa un algoritmo de búsqueda local para selección de imágenes.

    Args:
        data_dir: Diccionaro inicial que contiene las imágenes seleccionadas
        initial_percentage: Número que indica el porcentaje de las imágenes seleccionadas
        max_evaluations: Número máximo de evaluaciones
        max_evaluations_without_improvement: Criterio de parada si no hay mejoras
        metric: Métrica a optimizar ("accuracy" o "f1")
        model_name: Nombré del modelo a usar

    Returns:
        tuple: (best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done)
    """
    evaluations_without_improvement = 0
    evaluations_done = 0
    best_fitness = 0.0
    best_solution = {}
    best_fitness_history = []
    fitness_history = []

    while evaluations_done < max_evaluations:
        # Generar y evaluar solución
        current_solution = crear_dict_imagenes(data_dir, initial_percentage)
        current_fitness = fitness(dict_selection=current_solution, model_name=model_name, evaluations=evaluations_done)
        fitness_history.append(current_fitness)

        current_metric = current_fitness[metric.title()]

        evaluations_done += 1

        # Actualizar mejor solución global si corresponde
        if current_metric > best_fitness:
            best_fitness = current_metric
            best_solution = current_solution
            evaluations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en evaluación {evaluations_done}. Fitness: {best_fitness:.4f}")
        else:
            evaluations_without_improvement += 1

        best_fitness_history.append(best_fitness)

        # Criterio de parada por estancamiento
        if evaluations_without_improvement >= max_evaluations_without_improvement:
            print(f"Búsqueda terminada por estancamiento después de {evaluations_done} evaluaciones")
            break

    return best_solution, best_fitness, fitness_history, best_fitness_history, evaluations_done
