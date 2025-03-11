import argparse
import datetime
import os
import random
from pathlib import Path

import torch
import numpy as np
import polars as pl

from utils.classes import AlgorithmList
from utils.classes import DatasetList
from utils.classes import MetricList
from utils.classes import ModelList
from src.algorithms.genetic_algorithm import genetic_algorithm
from src.algorithms.genetic_algorithm2 import genetic_algorithm2
from src.algorithms.genetic_algorithm3 import genetic_algorithm_with_restart
from src.algorithms.local_search import local_search
from src.algorithms.memetic_algorithm import memetic_algorithm
from src.algorithms.random_search import random_search
from utils.utils import clear_ds_store
from utils.utils import fitness
from utils.utils import plot_boxplot
from utils.utils import plot_fitness_evolution
from utils.classes import ConfiguracionGlobal


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(task, date_path):
    if task == -1:
        task_path = ""
    else:
        task_path = "/task_" + task

    os.makedirs(f"img/{date_path}{task_path}", exist_ok=True)
    os.makedirs(f"logs/{date_path}", exist_ok=True)
    os.makedirs(f"results/csvs/{date_path}", exist_ok=True)
    os.makedirs(f"results/salidas/{date_path}", exist_ok=True)
    os.makedirs(f"tmp", exist_ok=True)


def main(
    initial_percentage: int = 10,
    max_evaluations: int = 10,
    max_evaluations_without_improvement: int = 10,
    algoritmo: str = "memetico",
    metric: str = "accuracy",
    model_name: str = "resnet"
):
    config = ConfiguracionGlobal()

    dataset = config.dataset

    create_directories(task=config.task_id, date_path=config.date)

    clear_ds_store(dataset)

    set_seed(24012000 + 1 + int(config.task_id))

    start = datetime.datetime.now()
    os.makedirs(f"logs/{config.date}", exist_ok=True)
    with open(f"logs/{config.date}/evaluations_log_{config.task_id}.txt", "a") as file:
        file.write(f"\n\n---------------------------------------"
                   f"{model_name}  {algoritmo.upper()}  {str(initial_percentage)}%-------"
                   f"---------------------------------------\n\n"
                   f"Start time: {str(start)}\n")
        file.flush()  # Forzar la escritura inmediata al disco

    print(f"\n\n--------------------------------------"
          f"{model_name}  {algoritmo.upper()}  {str(initial_percentage)}%-------"
          f"------------------------------------------")
    print("Start time: " + str(start))

    best_fitness = 0.0
    best_selection = {}
    fitness_history = []
    best_fitness_history = []
    evaluations_done = 0
    train_path = os.path.join(dataset, 'train')
    if algoritmo == "aleatorio":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = random_search(
            data_dir=train_path,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "busqueda local":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = local_search(
            data_dir=train_path,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            neighbor_size=10,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = genetic_algorithm(
            data_dir=train_path,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "memetico":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = memetic_algorithm(
            data_dir=train_path,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=3,
            mutation_rate=0.1,
            local_search_probability=0.2,
            local_search_evaluations=10,
            local_search_neighbor_size=5,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico2":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = genetic_algorithm2(
            data_dir=train_path,
            population_size=10,
            initial_percentage=initial_percentage,
            max_evaluations=max_evaluations,
            max_evaluations_without_improvement=max_evaluations_without_improvement,
            tournament_size=4,
            mutation_rate=0.05,
            metric=metric,
            model_name=model_name
        )
    elif algoritmo == "genetico3":
        best_selection, best_fitness, fitness_history, best_fitness_history, evaluations_done = (
            genetic_algorithm_with_restart(
                data_dir=train_path,
                population_size=10,
                initial_percentage=initial_percentage,
                max_evaluations=max_evaluations,
                max_evaluations_without_improvement=max_evaluations_without_improvement,
                tournament_size=4,
                mutation_rate=0.05,
                metric=metric,
                model_name=model_name
            ))

    end = datetime.datetime.now()
    duration = end - start
    duration = datetime.timedelta(seconds=int(duration.total_seconds()))

    print("End time: " + str(end))
    print("Duration: " + str(duration))
    print(f"\n\nMejor {metric} al acabar el algoritmo: {best_fitness:.4f}")

    if best_fitness != 0.0:
        print("\n\nFitness check:\n")
        os.makedirs("img/" + config.date, exist_ok=True)
        carpeta = f"img/{config.date}" + (f"/task_{config.task_id}" if config.task_id != -1 else "")
        os.makedirs(carpeta, exist_ok=True)
        # Crear y guardar la gráfica
        plot_fitness_evolution(
            fitness_history=best_fitness_history if max_evaluations != 1 else best_fitness_history * 50,
            initial_percentage=initial_percentage,
            algorithm_name=algoritmo,
            metric=metric,
            model=model_name,
            carpeta=carpeta
        )
        if algoritmo == "genetico3":
            plot_fitness_evolution(
                fitness_history=[fit[metric.title()] for fit in fitness_history] if max_evaluations != 1
                else [fit[metric.title()] for fit in fitness_history] * 50,
                initial_percentage=initial_percentage,
                algorithm_name=algoritmo,
                metric=metric,
                model=model_name,
                carpeta=carpeta
            )

        final_fitness = fitness(dict_selection=best_selection, model_name=model_name)
        print(f"\n\nMejor {metric} encontrado: {final_fitness[metric.title()]:.4f}")

        num_images = sum(1 for _ in Path(dataset).rglob('*') if _.is_file())
        images_selected = {key: value for key, value in best_selection.items() if value == 1}
        resultado = final_fitness | {
            "Duracion": str(duration),
            "Evaluaciones Realizadas": evaluations_done,
            "Porcentaje Final": len(images_selected) / num_images,
            "Porcentaje Paper": len(
                [key for key, value in images_selected.items() if 'paper' in key]
            ) / len(images_selected),
            "Porcentaje Rock": len(
                [key for key, value in images_selected.items() if 'rock' in key]
            ) / len(images_selected),
            "Porcentaje Scissors": len(
                [key for key, value in images_selected.items() if 'scissors' in key]
            ) / len(images_selected),
        }

        return resultado, fitness_history, best_fitness_history

    else:
        raise ValueError("No se ha seleccionado ningún algoritmo.")


if __name__ == "__main__":
    # Configuración de argumentos
    parser = argparse.ArgumentParser(description="Script de generación")
    parser.add_argument("--task_id", type=int, required=True, help="ID de la tarea para esta ejecución")
    algoritmo_input = input(
        "Introduce algoritmo para esta ejecución. Opciones: " + str([color.value for color in AlgorithmList])
    ) or 'busqueda local'
    metric_input = input(
        "Introduce metrica para esta ejecución. Opciones: " + str([color.value for color in MetricList])
    ) or 'accuracy'
    modelo_input = input(
        "Introduce modelo para esta ejecución. Opciones: " + str([color.value for color in ModelList])
    ) or 'mobilenet'
    dataset_input = input(
        "Introduce dataset para esta ejecución. Opciones: " + str([color.value for color in DatasetList])
    ) or 'rps'

    porcentaje_inicial = int(input(
        "Introduce dataset para esta ejecución. Default: 25"
    ) or 25)
    evaluaciones_maximas = int(input(
        "Introduce dataset para esta ejecución. Default: 10"
    ) or 10)
    evaluaciones_maximas_sin_mejora = int(input(
        "Introduce dataset para esta ejecución. Default: 10"
    ) or 10)
    task_id = parser.parse_args().task_id

    print(f"Task ID recibido: {task_id}")
    print(f"algoritmo recibido: {algoritmo_input}")
    print(f"metric recibido: {metric_input}")
    print(f"modelo recibido: {modelo_input}")
    print(f"dataset recibido: {dataset_input}")
    print(f"porcentaje_inicial recibido: {porcentaje_inicial}")
    print(f"evaluaciones_maximas recibido: {evaluaciones_maximas}")
    print(f"evaluaciones_maximas_sin_mejora recibido: {evaluaciones_maximas_sin_mejora}")
    print(f"---------------------------------------------------------\n")
    print(f"GPU: {torch.cuda.is_available()}")

    now = datetime.datetime.now()
    if os.getenv("SERVER") is not None:
        now = now + datetime.timedelta(hours=1)

    algoritmo: AlgorithmList = AlgorithmList(algoritmo_input.lower())
    metric: MetricList = MetricList(metric_input.lower())
    modelo: ModelList = ModelList(modelo_input.lower())
    dataset_choosen: DatasetList = DatasetList(dataset_input.upper())

    date = now.strftime("%Y-%m-%d_%H-%M")
    config = ConfiguracionGlobal(date=date, task_id=str(task_id), dataset=dataset_choosen.value)

    result, fitness_history, best_fitness_history = main(
        porcentaje_inicial,
        evaluaciones_maximas,
        evaluaciones_maximas_sin_mejora,
        algoritmo.value,
        metric.value,
        modelo.value
    )

    df = pl.DataFrame([result | {
        "Porcentaje Inicial": porcentaje_inicial / 100,
        "Algoritmo": algoritmo.value
    }], schema={
        "Algoritmo": pl.Utf8,
        "Porcentaje Inicial": pl.Float32,
        "Duracion": pl.Utf8,
        "Accuracy": pl.Float64,
        "Precision": pl.Float64,
        "Recall": pl.Float64,
        "F1-score": pl.Float64,
        "Evaluaciones Realizadas": pl.Int32,
        "Porcentaje Final": pl.Float32,
        "Porcentaje Paper": pl.Float32,
        "Porcentaje Rock": pl.Float32,
        "Porcentaje Scissors": pl.Float32
    })

    carpeta_img = f"img/{date}" + (f"/task_{task_id}" if task_id != -1 else "")
    try:
        # # ====== Boxplot 1: Fijando Algoritmo, variando Porcentaje Inicial ======
        # plot_boxplot(
        #     df=df,
        #     metric="Accuracy",  # O "Precision"
        #     eje_x="Porcentaje Inicial",
        #     hue=None,
        #     title="Comparación de Accuracy según Porcentaje Inicial y Algoritmo",
        #     filename=f'{carpeta_img}/{modelo.value}-BOXPLOT-accuracy-porcentaje.png',
        # )
        #
        # # ====== Boxplot 2: Fijando Porcentaje Inicial, variando Algoritmo ======
        # plot_boxplot(
        #     df=df,
        #     metric="Accuracy",  # O "Precision"
        #     eje_x="Algoritmo",
        #     hue=None,
        #     title="Comparación de Accuracy según Algoritmo y Porcentaje Inicial",
        #     filename=f'{carpeta_img}/{modelo.value}-BOXPLOT-accuracy-algoritmo.png',
        # )
        None
    except Exception as e:
        print("Ha fallado el bloxplot: " + str(e))

    df.write_csv(f"results/csvs/resultados_{date}_task_{task_id}.csv")