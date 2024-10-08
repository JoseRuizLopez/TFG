import datetime
import os
import random
import shutil
from typing import Literal

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def crear_dict_imagenes(data_dir: str, porcentaje_uso: int = 50):
    dict_dir = {}
    archivos = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            input_path = os.path.relpath(os.path.join(root, filename), data_dir)
            archivos.append(input_path)

    random.shuffle(archivos)
    num_usar = int(len(archivos) * (porcentaje_uso / 100))
    archivos_usar = archivos[:num_usar]

    for archivo in archivos:
        dict_dir[archivo] = 1 if archivo in archivos_usar else 0

    return dict_dir


def create_data_loaders(data_dir, dict_selection, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)

    # Filtrar el dataset basado en la selección JSON
    if dict_selection:
        indices = [i for i, (path, _) in enumerate(full_dataset.samples)
                   if dict_selection.get(os.path.relpath(path, data_dir), 0) == 1]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, data_loader, full_dataset.classes


def generate_neighbor(current_selection, neighbor_size, vary_percentage: bool = False):
    """
    Genera un vecino modificando la selección actual de imágenes.

    Args:
        current_selection (dict): Diccionario con nombres de imágenes como keys y 0/1 como values
        neighbor_size (int): Número de modificaciones a realizar
        vary_percentage (bool): Indica si debe de mantener el número de imágenes seleccionadas

    Returns:
        dict: Nueva selección de imágenes
    """
    # Crear una copia del diccionario actual
    neighbor = current_selection.copy()

    # Obtener listas de imágenes seleccionadas y no seleccionadas
    selected_images = [img for img, val in neighbor.items() if val == 1]
    unselected_images = [img for img, val in neighbor.items() if val == 0]

    # Realizar las modificaciones
    if vary_percentage:
        for _ in range(neighbor_size):
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
        for _ in range(int(neighbor_size/2)):
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
    data_dir: str = "dataset/train",
    initial_percentage: int = 10,
    max_iterations: int = 100,
    max_iterations_without_improvement: int = 20,
    neighbor_size: int = 10,
    metric: str = "accuracy",
    vary_percentage: bool = False
) -> tuple[dict, float]:
    """
    Implementa un algoritmo de búsqueda local para selección de imágenes.

    Args:
        data_dir: Diccionaro inicial que contiene las imágenes seleccionadas
        initial_percentage: Número que indica el porcentaje inicial de las imágenes seleccionadas
        max_iterations: Número máximo de iteraciones
        max_iterations_without_improvement: Criterio de parada si no hay mejoras
        neighbor_size: Número de cambios para generar cada vecino
        metric: Métrica a optimizar ("accuracy" o "f1")
        vary_percentage: Indica si el porcentaje de imágenes seleccionadas va a variar

    Returns:
        tuple: mejor_valor_fitness
    """
    # Generar solución inicial
    current_solution = crear_dict_imagenes(data_dir, initial_percentage)
    current_fitness = fitness(current_solution, metric)

    shutil.copy("best_checkpoint.pth", "best_model.pth")
    best_fitness = current_fitness
    best_solution = current_solution

    iterations_without_improvement = 0

    for iteration in range(max_iterations):
        # Generar y evaluar vecino
        neighbor = generate_neighbor(current_solution, neighbor_size, vary_percentage)
        neighbor_fitness = fitness(neighbor, metric)

        # Criterio de aceptación (se puede modificar para otras variantes)
        if neighbor_fitness >= current_fitness:
            current_solution = neighbor.copy()
            current_fitness = neighbor_fitness

            # Actualizar mejor solución global si corresponde
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution
                iterations_without_improvement = 0
                print(f"Nueva mejor solución encontrada en iteración {iteration}. Fitness: {best_fitness:.4f}")
            else:
                iterations_without_improvement += 1
        else:
            iterations_without_improvement += 1

        # Criterio de parada por estancamiento
        if iterations_without_improvement >= max_iterations_without_improvement:
            print(f"Búsqueda terminada por estancamiento en iteración {iteration}")
            break

    return best_solution, best_fitness


def random_search(
    data_dir: str = "dataset/train",
    initial_percentage: int = 10,
    max_iterations: int = 100,
    max_iterations_without_improvement: int = 20,
    metric: str = "accuracy"
) -> tuple[dict, float]:
    """
        Implementa un algoritmo de búsqueda local para selección de imágenes.

        Args:
            data_dir: Diccionaro inicial que contiene las imágenes seleccionadas
            initial_percentage: Número que indica el porcentaje de las imágenes seleccionadas
            max_iterations: Número máximo de iteraciones
            max_iterations_without_improvement: Criterio de parada si no hay mejoras
            metric: Métrica a optimizar ("accuracy" o "f1")

        Returns:
            tuple: mejor_valor_fitness
        """
    iterations_without_improvement = 0
    best_fitness = 0.0
    best_solution = {}

    for iteration in range(max_iterations):
        # Generar y evaluar vecino
        current_solution = crear_dict_imagenes(data_dir, initial_percentage)
        current_fitness = fitness(current_solution, metric)

        # Actualizar mejor solución global si corresponde
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = current_solution
            iterations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en iteración {iteration}. Fitness: {best_fitness:.4f}")
        else:
            iterations_without_improvement += 1

        # Criterio de parada por estancamiento
        if iterations_without_improvement >= max_iterations_without_improvement:
            print(f"Búsqueda terminada por estancamiento en iteración {iteration}")
            break

    return best_solution, best_fitness


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


def mutation(individual: dict, mutation_rate: float = 0.1) -> dict:
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


def tournament_selection(population: list[dict], fitness_values: list[float], tournament_size: int = 3) -> dict:
    """
    Selecciona un individuo mediante torneo.
    """
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_idx].copy()


def genetic_algorithm(
    data_dir: str = "dataset/train",
    population_size: int = 10,
    initial_percentage: int = 10,
    max_iterations: int = 50,
    max_iterations_without_improvement: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    metric: str = "accuracy"
) -> tuple[dict, float]:
    """
    Implementa un algoritmo genético para la selección de imágenes.

    Args:
        data_dir: Directorio con las imágenes
        population_size: Tamaño de la población
        initial_percentage: Porcentaje inicial de imágenes a seleccionar
        max_iterations: Número máximo de generaciones
        max_iterations_without_improvement: Criterio de parada si no hay mejoras
        tournament_size: Tamaño del torneo para selección
        mutation_rate: Probabilidad de mutación
        metric: Métrica a optimizar ("accuracy" o "f1")

    Returns:
        tuple: mejor_valor_fitness
    """
    # Generar población inicial
    population = [crear_dict_imagenes(data_dir, initial_percentage)
                  for _ in range(population_size)]

    # Evaluar población inicial
    fitness_values = [fitness(ind, metric) for ind in population]

    # Encontrar el mejor individuo inicial
    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    shutil.copy("best_checkpoint.pth", "best_model.pth")

    iterations_without_improvement = 0

    # Bucle principal del algoritmo genético
    for generation in range(max_iterations):
        # Crear nueva población
        new_population = []
        new_fitness_values = []

        # Elitismo: mantener al mejor individuo
        new_population.append(best_individual.copy())
        new_fitness_values.append(best_fitness)

        # Generar el resto de la población
        while len(new_population) < population_size:
            # Selección
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            # Cruce
            child1, child2 = crossover(parent1, parent2)

            # Mutación
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            # Evaluar y añadir hijos
            for child in [child1, child2]:
                if len(new_population) < population_size:
                    child_fitness = fitness(child, metric)
                    new_population.append(child)
                    new_fitness_values.append(child_fitness)

        # Actualizar población
        population = new_population
        fitness_values = new_fitness_values

        # Actualizar mejor solución
        current_best_idx = fitness_values.index(max(fitness_values))
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            iterations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en generación {generation}. Fitness: {best_fitness:.4f}")
        else:
            iterations_without_improvement += 1

        # Criterio de parada
        if iterations_without_improvement >= max_iterations_without_improvement:
            print(f"Búsqueda terminada por estancamiento en generación {generation}")
            break

    return best_individual, best_fitness


def local_search_improvement(
    individual: dict,
    metric: str,
    max_iterations: int = 10,
    neighbor_size: int = 5
) -> tuple[dict, float]:
    """
    Mejora un individuo mediante búsqueda local.
    """
    current_solution = individual.copy()
    current_fitness = fitness(current_solution, metric)

    best_solution = current_solution.copy()
    best_fitness = current_fitness

    for _ in range(max_iterations):
        # Generar y evaluar vecino
        neighbor = generate_neighbor(current_solution, neighbor_size)
        neighbor_fitness = fitness(neighbor, metric)

        # Criterio de aceptación
        if neighbor_fitness > current_fitness:
            current_solution = neighbor.copy()
            current_fitness = neighbor_fitness

            if current_fitness > best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness

    return best_solution, best_fitness


def memetic_algorithm(
    data_dir: str = "dataset/train",
    population_size: int = 20,
    initial_percentage: int = 10,
    max_iterations: int = 50,
    max_iterations_without_improvement: int = 10,
    tournament_size: int = 3,
    mutation_rate: float = 0.1,
    local_search_probability: float = 0.2,
    local_search_iterations: int = 10,
    local_search_neighbor_size: int = 5,
    metric: str = "accuracy"
) -> tuple[dict, float]:
    """
    Implementa un algoritmo memético para la selección de imágenes.

    Args:
        data_dir: Directorio con las imágenes
        population_size: Tamaño de la población
        initial_percentage: Porcentaje inicial de imágenes a seleccionar
        max_iterations: Número máximo de generaciones
        max_iterations_without_improvement: Criterio de parada si no hay mejoras
        tournament_size: Tamaño del torneo para selección
        mutation_rate: Probabilidad de mutación
        local_search_probability: Probabilidad de aplicar búsqueda local a un individuo
        local_search_iterations: Número de iteraciones de búsqueda local
        local_search_neighbor_size: Tamaño del vecindario en búsqueda local
        metric: Métrica a optimizar ("accuracy" o "f1")

    Returns:
        tuple: (mejor_solucion, mejor_valor_fitness)
    """
    # Generar población inicial
    population = [crear_dict_imagenes(data_dir, initial_percentage)
                  for _ in range(population_size)]

    # Evaluar población inicial
    fitness_values = [fitness(ind, metric) for ind in population]

    # Encontrar el mejor individuo inicial
    best_fitness_idx = fitness_values.index(max(fitness_values))
    best_individual = population[best_fitness_idx].copy()
    best_fitness = fitness_values[best_fitness_idx]
    shutil.copy("best_checkpoint.pth", "best_model.pth")

    iterations_without_improvement = 0

    # Bucle principal del algoritmo memético
    for generation in range(max_iterations):
        print(f"\nGeneración {generation + 1}/{max_iterations}")

        # Crear nueva población
        new_population = []
        new_fitness_values = []

        # Elitismo: mantener al mejor individuo
        new_population.append(best_individual.copy())
        new_fitness_values.append(best_fitness)

        # Generar el resto de la población
        while len(new_population) < population_size:
            # Selección
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)

            # Cruce
            child1, child2 = crossover(parent1, parent2)

            # Mutación
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)

            # Mejorar mediante búsqueda local si aplica
            for child in [child1, child2]:
                if len(new_population) < population_size:
                    # Aplicar búsqueda local con cierta probabilidad
                    if random.random() < local_search_probability:
                        print("Aplicando búsqueda local a un individuo...")
                        improved_child, child_fitness = local_search_improvement(
                            child,
                            metric,
                            local_search_iterations,
                            local_search_neighbor_size
                        )
                        new_population.append(improved_child)
                    else:
                        child_fitness = fitness(child, metric)
                        new_population.append(child)
                    new_fitness_values.append(child_fitness)

        # Actualizar población
        population = new_population
        fitness_values = new_fitness_values

        # Actualizar mejor solución
        current_best_idx = fitness_values.index(max(fitness_values))
        if fitness_values[current_best_idx] > best_fitness:
            best_individual = population[current_best_idx].copy()
            best_fitness = fitness_values[current_best_idx]
            iterations_without_improvement = 0
            print(f"Nueva mejor solución encontrada en generación {generation}. Fitness: {best_fitness:.4f}")
            shutil.copy("best_checkpoint.pth", "best_model.pth")
        else:
            iterations_without_improvement += 1

        print(f"Mejor fitness actual: {best_fitness:.4f}")
        print(f"Generaciones sin mejora: {iterations_without_improvement}")

        # Criterio de parada
        if iterations_without_improvement >= max_iterations_without_improvement:
            print(f"Búsqueda terminada por estancamiento en generación {generation}")
            break

    return best_fitness


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()   # Poner a cero los gradientes de los parametros gestionados por el optimizador.
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), "best_checkpoint.pth")
            print("Model saved!")


def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return accuracy, precision, recall, f1


def fitness(dict_selection: dict, metric: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear data loaders
    train_dataset, train_loader, train_classes = create_data_loaders("dataset/train", dict_selection)

    # Los nombres de este dataset son erroneos. El test y el valid están intercambiados.
    valid_dataset, valid_loader, _ = create_data_loaders("dataset/test", None)  # No filtramos el conjunto de validación
    test_dataset, test_loader, _ = create_data_loaders("dataset/valid", None)  # No filtramos el conjunto de prueba

    # Definir el modelo
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Congelar todas las capas
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la última capa fully connected
    num_classes = len(train_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Descongelar la última capa
    for param in model.fc.parameters():
        param.requires_grad = True

    model = model.to(device)

    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    # Modificar el optimizador para que solo actualice los parámetros de la última capa
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Entrenar el modelo
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device=device)

    # Cargar el mejor modelo
    model.load_state_dict(torch.load("best_checkpoint.pth"))

    # Evaluar el modelo
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device=device)

    if metric == "accuracy":
        return accuracy
    else:
        return f1


def main(
    algoritmo: Literal["aleatorio", "busqueda local", "genetico", "memetico"] = "memetico",
    metric: Literal["accuracy", "f1"] = "accuracy"
):
    seed = 24012000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = "dataset/train"
    initial_percentage = 10
    max_iterations = 10
    max_iterations_without_improvement = 10

    start = datetime.datetime.now()
    print(f"\n\n--------------------------------------"
          f"----------------{algoritmo.upper()}-------"
          f"------------------------------------------")
    print("Start time: " + str(start))

    if algoritmo == "aleatorio":
        best_selection, best_fitness = random_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_iterations=max_iterations,
            max_iterations_without_improvement=max_iterations_without_improvement,
            metric=metric
        )
    elif algoritmo == "busqueda local":
        best_selection, best_fitness = local_search(
            data_dir=dataset,
            initial_percentage=initial_percentage,
            max_iterations=max_iterations,
            max_iterations_without_improvement=max_iterations_without_improvement,
            neighbor_size=10,
            metric=metric
        )
    elif algoritmo == "genetico":
        best_selection, best_fitness = genetic_algorithm(
            data_dir=dataset,
            population_size=10,
            initial_percentage=initial_percentage,
            max_iterations=max_iterations,
            max_iterations_without_improvement=max_iterations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            metric=metric
        )
    elif algoritmo == "memetico":
        best_selection, best_fitness = memetic_algorithm(
            data_dir=dataset,
            population_size=10,
            initial_percentage=initial_percentage,
            max_iterations=max_iterations,
            max_iterations_without_improvement=max_iterations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            local_search_probability=0.2,
            local_search_iterations=10,
            local_search_neighbor_size=5,
            metric=metric
        )
    else:
        best_fitness = 0.0
        best_selection = {}

    end = datetime.datetime.now()
    print("End time: " + str(end))
    print("Duration: " + str(end - start))

    if best_fitness != 0.0:
        final_fitness = fitness(best_selection, metric)
        print(f"\n\nMejor {metric} encontrado: {final_fitness:.4f}")
    else:
        print("No se ha seleccionado ningún algoritmo.")


if __name__ == "__main__":
    main("aleatorio", "accuracy")
    main("busqueda local", "accuracy")
    main("genetico", "accuracy")
    main("memetico", "accuracy")

