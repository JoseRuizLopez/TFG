import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def crear_json_imagenes(data_dir, porcentaje_uso=50):
    lista_dir = {}
    archivos = []

    for root, _, files in os.walk(data_dir):
        for filename in files:
            input_path = os.path.relpath(os.path.join(root, filename), data_dir)
            archivos.append(input_path)

    random.shuffle(archivos)
    num_usar = int(len(archivos) * (porcentaje_uso / 100))
    archivos_usar = archivos[:num_usar]

    for archivo in archivos:
        lista_dir[archivo] = 1 if archivo in archivos_usar else 0

    with open('lista_de_imagenes.json', 'w') as archivo:
        json.dump(lista_dir, archivo, indent=2)

    return lista_dir


def leer_json_imagenes():
    with open('lista_de_imagenes.json', 'r') as archivo:
        return json.load(archivo)


def create_data_loaders(data_dir, json_selection, batch_size=32):
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
    if json_selection:
        indices = [i for i, (path, _) in enumerate(full_dataset.samples)
                   if json_selection.get(os.path.relpath(path, data_dir), 0) == 1]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, data_loader, full_dataset.classes


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

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
            torch.save(model.state_dict(), "best_model.pth")
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


def main(porcentaje_uso=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Crear o leer el JSON de selección de imágenes
    json_selection = crear_json_imagenes("dataset/train", porcentaje_uso)

    # Crear data loaders
    train_dataset, train_loader, train_classes = create_data_loaders("dataset/train", json_selection)
    valid_dataset, valid_loader, _ = create_data_loaders("dataset/valid",
                                                         None)  # No filtramos el conjunto de validación
    test_dataset, test_loader, _ = create_data_loaders("dataset/test", None)  # No filtramos el conjunto de prueba

    # Definir el modelo
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = len(train_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Definir criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar el modelo
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, device=device)

    # Cargar el mejor modelo
    model.load_state_dict(torch.load("best_model.pth"))

    # Evaluar el modelo
    accuracy, precision, recall, f1 = evaluate_model(model, test_loader, device=device)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    resultados = {}
    # porcentajes = [50]
    porcentajes = [10, 25, 50, 75, 100]
    for porcentaje in porcentajes:
        print(f"\nEntrenando con {porcentaje}% de las imágenes:")
        accuracy, precision, recall, f1 = main(porcentaje)
        resultados[porcentaje] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    print("\nResumen de resultados:")
    for porcentaje, metricas in resultados.items():
        print(f"{porcentaje}% de imágenes: Accuracy={metricas['accuracy']:.4f}, F1={metricas['f1']:.4f}")
