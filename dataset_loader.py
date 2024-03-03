import json
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image, ExifTags


# Función para detectar las imagenes seleccionadas usando su path/nombre
def is_valid_file_func1(file_path):
    return file_path.lower().endswith(('.png', '.jpg', '.jpeg'))

def is_valid_file_func2(file_path):
    try:
        with Image.open(file_path) as img:
            try:
                exif_data = img._getexif()
            except AttributeError:
                exif_data = None
            
            # Mostrar los metadatos Exif originales
            print("Metadatos Exif originales:")
            if exif_data:
                for tag, value in exif_data.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    print(f"{tag_name}: {value}")
            else:
                print("No hay metadatos Exif disponibles.")
                
            return True
    except Exception as e:
        print(f"Error al procesar la imagen {file_path}: {e}")
        return False


# Función para detectar las imagenes seleccionadas usando metadatos con imágenes TIFF
def is_valid_file_func3(file_path):
    try:
        with Image.open(file_path) as img:
            # Acceder a los metadatos personalizados (tipo 270)
            custom_metadata_tag = 270
            
            if custom_metadata_tag in img.tag_v2:
                custom_metadata_json = img.tag_v2[custom_metadata_tag]
                custom_metadata = json.loads(custom_metadata_json)
                print(f"Metadatos personalizados para {file_path}: {custom_metadata}")
            else:
                print(f"No se encontraron metadatos personalizados para {file_path}")
                
            return custom_metadata['Seleccionado']
    except Exception as e:
        print(f"Error al procesar la imagen {file_path}: {e}")
        return False


def leer_dataset():
    data_dir = "dataset_mod/train/"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Cambia el tamaño de la imagen a 256x256 píxeles
        transforms.ToTensor(),           # Convierte la imagen a un tensor
    ])

    dataset = ImageFolder(root=data_dir, transform=transform, is_valid_file=is_valid_file_func3)

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterar sobre el DataLoader para acceder a los lotes de datos
    for batch_idx, (images, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}: Images shape {images.shape}, Labels shape {labels.shape}")

def reguardar_imagenes():
    data_dir_original = "dataset/"
    data_dir_mod = "dataset_mod/"
    
    os.makedirs(data_dir_mod, exist_ok=True)

    for root, _, files in os.walk(data_dir_original):
        for filename in files:
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                input_path = os.path.join(root, filename)

                relative_path = os.path.relpath(input_path, data_dir_original)
                output_path = os.path.join(data_dir_mod, os.path.splitext(relative_path)[0] + '.tiff')

                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                img = Image.open(input_path)

                custom_metadata = {"Seleccionado": 1}
                
                img.save(output_path, format='TIFF', tiffinfo={270: json.dumps(custom_metadata)})
                
                img.close()


# reguardar_imagenes()
leer_dataset()
