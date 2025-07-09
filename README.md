

# Memetic algorithms for reducing training data in deep convolutional learning models

This repository contains the code, experiments, and documentation for the Bachelor's Thesis (TFG) titled **"Memetic algorithms for reducing training data in deep convolutional learning models"** by José Ruiz López at the University of Granada.

## Description

The main goal of this thesis is to investigate and compare metaheuristic techniques (random, local search, genetic, and memetic) for the intelligent selection of image subsets in deep learning tasks. The aim is to reduce the size of training datasets without significant loss of performance, improving computational efficiency and model sustainability.

Different variants of evolutionary algorithms have been implemented and evaluated on several image datasets, using reference models such as ResNet50 and MobileNetV2.

## Repository structure

- **src/**: Main source code for algorithms and utilities.
- **data/**: Datasets used in the experiments (CIFAR10, PAINTING, RPS).
- **results/**: Experimental results and generated tables.
- **img/**: Images generated during the experiments.
- **logs/**: Execution logs and intermediate results.
- **scripts/**: Scripts to automate experiments and generate plots.
- **docs/**: Documentation in LaTeX, chapters, figures, and bibliography.
- **requirements.txt**: Project dependencies.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/JoseRuizLopez/TFG.git
   cd TFG
   ```

2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

The main scripts to run the algorithms and generate results are located in the [`scripts/`](scripts/) folder. Check the internal documentation and code comments for usage details.

## Datasets

The experiments were conducted on the following datasets:
- **Rock, Paper, Scissors**
- **PAINTING (Art Images)**
- **CIFAR-10**

See [`docs/capitulos/07_Entorno_Experimental.tex`](docs/capitulos/07_Entorno_Experimental.tex) for more details on data structure and preprocessing.

## Documentation

The full thesis document is available at [`docs\out\proyecto.pdf`](docs\out\proyecto.pdf) and its associated chapters.

## License

This project is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license for datasets and open source for the rest of the components.

## Author

- José Ruiz López  
- Advisor: Daniel Molina Cabrera  
- University of Granada

---

Original repository: [https://github.com/JoseRuizLopez/TFG](https://github.com/JoseRuizLopez/TFG)

---

# Algoritmos meméticos para reducir datos de entrenamiento en modelos de aprendizaje profundo convolucionales

Este repositorio contiene el código, experimentos y documentación del Trabajo de Fin de Grado (TFG) titulado **"Algoritmos meméticos para reducir datos de entrenamiento en modelos de aprendizaje profundo convolucionales"** realizado por José Ruiz López en la Universidad de Granada.

## Descripción

El objetivo principal de este TFG es investigar y comparar técnicas metaheurísticas (aleatorias, búsqueda local, genéticas y meméticas) para la selección inteligente de subconjuntos de imágenes en tareas de aprendizaje profundo. El fin es reducir el tamaño de los conjuntos de entrenamiento sin pérdida significativa de rendimiento, mejorando la eficiencia computacional y la sostenibilidad de los modelos.

Se han implementado y evaluado diferentes variantes de algoritmos evolutivos sobre varios datasets de imágenes, utilizando modelos de referencia como ResNet50 y MobileNetV2.

## Estructura del repositorio

- **src/**: Código fuente principal de los algoritmos y utilidades.
- **data/**: Datasets utilizados en los experimentos (CIFAR10, PAINTING, RPS).
- **results/**: Resultados experimentales y tablas generadas.
- **img/**: Imágenes generadas durante los experimentos.
- **logs/**: Registros de ejecución y resultados intermedios.
- **scripts/**: Scripts para automatizar experimentos y generación de gráficos.
- **docs/**: Documentación en LaTeX, capítulos, figuras y bibliografía.
- **requirements.txt**: Dependencias del proyecto.

## Instalación

1. Clona el repositorio:
   ```sh
   git clone https://github.com/JoseRuizLopez/TFG.git
   cd TFG
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```sh
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Uso

Los scripts principales para ejecutar los algoritmos y generar resultados se encuentran en la carpeta [`scripts/`](scripts/). Consulta la documentación interna y los comentarios en el código para detalles sobre su uso.

## Datasets

Los experimentos se han realizado sobre los siguientes conjuntos de datos:
- **Rock, Paper, Scissors**
- **PAINTING (Art Images)**
- **CIFAR-10**

Consulta [`docs/capitulos/07_Entorno_Experimental.tex`](docs/capitulos/07_Entorno_Experimental.tex) para más detalles sobre la estructura y preprocesamiento de los datos.

## Documentación

La memoria completa del TFG está disponible en [`docs\out\proyecto.pdf`](docs\out\proyecto.pdf) y sus capítulos asociados.

## Licencia

Este proyecto se distribuye bajo licencia Creative Commons Attribution 4.0 International (CC BY 4.0) para los datasets y código abierto para el resto de componentes.

## Autor

- José Ruiz López  
- Tutor: Daniel Molina Cabrera  
- Universidad de Granada

---

Repositorio original: [https://github.com/JoseRuizLopez/TFG](https://github.com/JoseRuizLopez/TFG)
