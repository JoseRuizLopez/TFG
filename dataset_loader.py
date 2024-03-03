import json
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

    # Iterar sobre el DataLoader para acceder a los lotes de datos
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}: Images shape {images.shape}, Labels shape {labels.shape}")



# lista_dir = crear_json_imagenes()
lista_imagenes = leer_json_imagenes()


leer_dataset(lista_imagenes=lista_imagenes)
