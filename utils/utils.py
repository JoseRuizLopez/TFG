import datetime
import os
import random
import shutil
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from torch.utils.data import random_split
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V2_Weights
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.classes import ConfiguracionGlobal


def plot_fitness_evolution(
    fitness_history: List[float],
    initial_percentage: int,
    algorithm_name: str,
    metric: str,
    model: str,
    carpeta: str
):
    """
    Crea y guarda una gráfica que muestra la evolución del fitness.

    Args:
        fitness_history: Lista con los valores de fitness
        initial_percentage: Entero con el porcentaje inicial de imagenes seleccionadas
        algorithm_name: Nombre del algoritmo utilizado
        metric: Métrica utilizada (accuracy o f1)
        model: Nombre del modelo usado
        carpeta: Nombre de la carpeta
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o')
    plt.title(f'Evolución del {metric} - Algoritmo {algorithm_name} - Modelo {model} - '
              f'Porcentaje Inicial {str(initial_percentage)}%')
    plt.xlabel('Iteración')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.savefig(f'{carpeta}/{model}-{algorithm_name.replace(" ", "_")}-{str(initial_percentage)}-{metric}.png')
    plt.close()


def plot_multiple_fitness_evolution(
    data: List[List[float]],
    labels: List[str],
    metric: str,
    title: str,
    filename: str,
    x_label: str = "Iteración"
):
    """
    Crea y guarda una gráfica que muestra la evolución del fitness multiple.

    Args:
        data: Lista de listas, donde cada lista interna contiene los valores de una línea a lo largo de las iteraciones
        labels: Lista de etiquetas para cada línea
        metric: Métrica utilizada (accuracy o f1)
        title: Título de la gráfica
        filename: Nombre del archivo generado en el directorio
        x_label: Nombre de la etiqueta para el eje x
    """
    # Encontrar la longitud máxima entre todas las listas
    max_length = max(len(lst) for lst in data)
    max_length = max_length if max_length != 1 else 50

    # Extender cada lista repitiendo el último valor hasta la longitud máxima
    extended_data = []
    for lst in data:
        extended_lst = lst + [lst[-1]] * (max_length - len(lst))
        extended_data.append(extended_lst)

    plt.figure(figsize=(10, 6))

    # Graficar cada lista interna de data
    for i, line_data in enumerate(extended_data):
        plt.plot(line_data, label=labels[i])

    # Títulos y etiquetas
    plt.title(title, fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(metric.capitalize(), fontsize=12)

    # Mostrar la leyenda
    plt.legend(loc='best', fontsize=10)

    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_min_max_lines(df: pd.DataFrame | pl.DataFrame, y_col, x_col):
    # Convertir a pandas si es un DataFrame de polars (porque matplotlib funciona mejor con pandas)
    if isinstance(df, pl.DataFrame):
        # df = df.to_pandas()  # No disponible por la version de polars
        df = pd.DataFrame(df.to_dict())

    # Obtener los valores mínimo y máximo por cada categoría en x_col
    grouped = df.groupby(x_col)[y_col].agg(['min', 'max'])

    # Iterar sobre cada categoría
    for i, (cat, values) in enumerate(grouped.iterrows()):
        min_val, max_val = values['min'], values['max']

        # Dibujar líneas horizontales en los valores min y max
        plt.plot([i - 0.2, i + 0.2], [min_val, min_val], color='red', lw=2)
        plt.plot([i - 0.2, i + 0.2], [max_val, max_val], color='blue', lw=2)

        # Anotar los valores exactos
        plt.text(i, min_val, f'{min_val:.3f}', ha='center', va='top', fontsize=10, color='red')
        plt.text(i, max_val, f'{max_val:.3f}', ha='center', va='bottom', fontsize=10, color='blue')


def plot_boxplot(df: pd.DataFrame, metric: str, filename: str | None, hue: str | None, title: str, eje_x: str):

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x=eje_x, y=metric.title())

    plt.title(title)
    plt.xlabel(eje_x)
    plt.ylabel(metric.title())
    if hue:
        plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='best')

    plot_min_max_lines(df, metric, eje_x)

    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)

    plt.show()


def create_data_loaders(data_dir, dict_selection, weights, batch_size=32):
    transform = weights.transforms()

    full_dataset = ImageFolder(
        root=data_dir,
        transform=transform,
        is_valid_file=is_valid_image  # Filtra archivos no válidos
    )

    # Filtrar el RPS basado en la selección JSON
    if dict_selection:
        indices = [i for i, (path, _) in enumerate(full_dataset.samples)
                   if dict_selection.get(os.path.relpath(path, data_dir), 0) == 1]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, data_loader, full_dataset.classes


# Verifica que el archivo tenga una extensión válida
def is_valid_image(file_path):
    valid_extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
    return os.path.splitext(file_path)[1].lower() in valid_extensions


def generate_datasets(dataset_path, weights, dict_selection=None, batch_size=32, train_split=0.8):
    """
    Carga los datasets de train, valid y test.
    Si no existe valid, lo crea a partir del train dataset.
    Aplica un filtro en train basado en dict_selection.

    Args:
        dataset_path (str): Ruta base donde están las carpetas train, valid y test.
        weights: Transformaciones predefinidas de torchvision.
        dict_selection (dict, opcional): Diccionario con imágenes seleccionadas para train.
        batch_size (int): Tamaño del batch para los DataLoaders.
        train_split (float): Proporción del train dataset que se usará para entrenamiento (default 80%).

    Returns:
        dict: Diccionario con los DataLoaders de train, valid y test.
    """

    # Cargar el dataset completo desde la carpeta de entrenamiento
    full_train_dataset = ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=weights.transforms(),
        is_valid_file=is_valid_image  # Filtra archivos no válidos
    )
    classes = full_train_dataset.classes  # Obtener las clases del dataset

    # Aplicar filtro basado en `dict_selection`
    if dict_selection:
        indices = [i for i, (path, _) in enumerate(full_train_dataset.samples)
                   if dict_selection.get(os.path.relpath(path, os.path.join(dataset_path, "train")), 0) == 1]
        filtered_train_dataset = Subset(full_train_dataset, indices)
    else:
        filtered_train_dataset = full_train_dataset  # Si no hay filtro, usar dataset completo

    # Comprobar si la carpeta de validación existe
    valid_path = os.path.join(dataset_path, "valid")

    if not os.path.exists(valid_path):
        print("No se encontró la carpeta de validación. Creando valid set desde train...")

        train_size = int(train_split * len(filtered_train_dataset))  # 80% train
        valid_size = len(filtered_train_dataset) - train_size  # 20% valid

        train_dataset, valid_dataset = random_split(filtered_train_dataset, [train_size, valid_size])
    else:
        print("Carpeta de validación encontrada. Cargando valid dataset...")

        train_dataset = filtered_train_dataset  # Train se mantiene con el filtro aplicado
        valid_dataset = ImageFolder(
            root=valid_path,
            transform=weights.transforms(),
            is_valid_file=is_valid_image  # Filtra archivos no válidos
        )

    # Cargar el test dataset
    test_dataset = ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=weights.transforms(),
        is_valid_file=is_valid_image  # Filtra archivos no válidos
    )

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Datasets cargados correctamente.")

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
    }, classes


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    best_valid_loss = float('inf')

    print(f"Using device: {device}")  # Añadido para verificar el dispositivo
    model = model.to(device)  # Asegurar que el modelo está en GPU

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Mover los datos a GPU
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                # Mover los datos a GPU
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Valid Loss: {avg_valid_loss:.4f}")

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            # Guardar el modelo en CPU para evitar problemas de compatibilidad
            model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            config = ConfiguracionGlobal()
            torch.save(model_state, f"tmp/best_checkpoint{config.task_id}.pth")
            print("Model saved!")


def evaluate_model(model, test_loader, device):
    model = model.to(device)  # Asegurar que el modelo está en GPU
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Mover los datos a GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Mover predicciones y labels a CPU para métricas
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


def fitness(dict_selection: dict, model_name: str = "resnet", evaluations: int | None = None):
    config = ConfiguracionGlobal()

    dataset_path = config.dataset

    old_seed = torch.seed()
    seed = 5234 + int(config.task_id)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Verificar y mostrar la disponibilidad de GPU
    if os.getenv("SERVER") is not None:
        while torch.cuda.device_count() < 1:
            pass
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    # Definir el modelo
    if model_name == "resnet":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        weights = MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)

    # He renombrado el dataset PAINTING para que concidan los nonmbres
    # He renombrado el dataset RPS, cambiando el test por test y el test por test

    # Crear data loaders
    loaders, classes = generate_datasets(
        dataset_path=dataset_path,
        weights=weights,
        dict_selection=dict_selection,
        batch_size=32,
        train_split=0.8
    )

    train_loader = loaders["train_loader"]
    valid_loader = loaders["valid_loader"]
    test_loader = loaders["test_loader"]

    # Congelar todas las capas
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la última capa fully connected
    num_classes = len(classes)
    if model_name == "resnet":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, num_classes)
        )

    # Descongelar la última capa
    if model_name == "resnet":
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.classifier.parameters():
            param.requires_grad = True

    model = model.to(device)

    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss().to(device)  # Mover criterion a GPU
    if model_name == "resnet":
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Entrenar el modelo
    train_model(model, train_loader, valid_loader, criterion, optimizer, device=device, num_epochs=10)

    # Cargar el mejor modelo
    checkpoint = torch.load(f"tmp/best_checkpoint{config.task_id}.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    # Evaluar el modelo
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device=device)

    # Liberar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memory Allocated after cleanup: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    torch.manual_seed(old_seed)
    torch.cuda.manual_seed_all(old_seed)

    if evaluations is not None:
        with open(f"logs/{config.date}/evaluations_log_{config.task_id}.txt", "a") as file:
            file.write(f"Evaluación {str(evaluations+1)} -> {datetime.datetime.now()}\n")
            file.flush()  # Forzar la escritura inmediata al disco

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }


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


def clear_ds_store(dataset_path: str):
    # Recorrer el dataset y eliminar .DS_Store (archivos y carpetas)
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if name == ".DS_Store":
                file_path = os.path.join(root, name)
                os.remove(file_path)
                print(f"Archivo eliminado: {file_path}")

        for name in dirs:
            if name == ".DS_Store":
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path, ignore_errors=True)
                print(f"Directorio eliminado: {dir_path}")
