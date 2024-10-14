import datetime
import os
import random
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import MobileNet_V2_Weights
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def plot_fitness_evolution(
    fitness_history: List[float],
    initial_percentage: int,
    algorithm_name: str,
    metric: str,
    model: str
):
    """
    Crea y guarda una gráfica que muestra la evolución del fitness.

    Args:
        fitness_history: Lista con los valores de fitness
        initial_percentage: Entero con el porcentaje inicial de imagenes seleccionadas
        algorithm_name: Nombre del algoritmo utilizado
        metric: Métrica utilizada (accuracy o f1)
        model: Nombre del modelo usado
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, marker='o')
    plt.title(f'Evolución del {metric} - Algoritmo {algorithm_name} - Modelo {model} - '
              f'Porcentaje Inicial {str(initial_percentage)}%')
    plt.xlabel('Iteración')
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.savefig(f'img/{model}-{algorithm_name}-{str(initial_percentage)}-{metric}.png')
    plt.close()


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
            torch.save(model_state, "results/best_checkpoint.pth")
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


def fitness(dict_selection: dict, metric: str, model_name: str = "resnet", evaluations: int | None = None):
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

    # Crear data loaders
    train_dataset, train_loader, train_classes = create_data_loaders("data/dataset/train", dict_selection)
    valid_dataset, valid_loader, _ = create_data_loaders("data/dataset/test", None)
    test_dataset, test_loader, _ = create_data_loaders("data/dataset/valid", None)

    # Definir el modelo
    if model_name == "resnet":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Congelar todas las capas
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar la última capa fully connected
    num_classes = len(train_classes)
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
    checkpoint = torch.load("results/best_checkpoint.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    # Evaluar el modelo
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device=device)

    # Liberar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Memory Allocated after cleanup: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

    if evaluations is not None:
        with open("results/evaluations_logs.txt", "a") as file:
            file.write(f"Evaluación {str(evaluations+1)} -> {str(datetime.datetime.now())}\n")
            file.flush()  # Forzar la escritura inmediata al disco

    else:
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        }

    if metric == "accuracy":
        return accuracy
    else:
        return f1


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
