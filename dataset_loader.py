import json
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights


def crear_json_imagenes():
    data_dir = "dataset/train"
    lista_dir = {}

    for root, _, files in os.walk(data_dir):
        for filename in files:
            input_path = os.path.join(root, filename)
            lista_dir[input_path] = 1
            
    
    with open('lista_de_imagenes.json', 'w') as archivo:
        json.dump(lista_dir, archivo, indent=2) 
    
    return lista_dir
    

def leer_json_imagenes():
    nombre_json = 'lista_de_imagenes.json'
    
    with open(nombre_json, 'r') as archivo:
        lista_imagenes = json.load(archivo)
        
    return lista_imagenes


# Función para detectar las imagenes seleccionadas usando su path/nombre
def is_valid_file_func(file_path):
    return lista_imagenes[file_path]


def leer_dataset(lista_imagenes):
    data_dir = "dataset/train/"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Cambia el tamaño de la imagen a 256x256 píxeles
        transforms.ToTensor(),           # Convierte la imagen a un tensor
    ])

    dataset = ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file_func)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataset, data_loader


def entrenar_modelo(dataset, data_loader, use_gpu=True):
    # Verificar si hay una GPU disponible y configurar el dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    # Definir el modelo ResNet-50 con pesos preentrenados
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Mover el modelo al dispositivo (CPU o GPU)
    model = model.to(device)

    # Modificar la última capa (clasificación) para adaptarse al conjunto de datos
    num_classes = len(dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Definir la función de pérdida y el optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Establecer el modelo en modo de entrenamiento

        for batch_idx, (images, labels) in enumerate(data_loader):
            # Mover los datos al dispositivo
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reiniciar los gradientes

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calcular la pérdida
            loss.backward()  # Backward pass
            optimizer.step()  # Actualizar los pesos

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item()}")

    print("Entrenamiento completado.")
    
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), "modelo_entrenado.pth")



lista_dir = crear_json_imagenes()
lista_imagenes = leer_json_imagenes()

dataset, data_loader = leer_dataset(lista_imagenes=lista_imagenes)

entrenar_modelo(dataset, data_loader, use_gpu=True)
