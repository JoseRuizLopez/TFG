from enum import Enum


class AlgorithmList(Enum):
    # ALEATORIO = "aleatorio"
    # BUSQUEDA_LOCAL = "busqueda local"
    FREE_BUSQUEDA_LOCAL = "free busqueda local"
    # GENETICO = "genetico"
    # MEMETICO = "memetico"
    # GENETICO2 = "genetico2"
    GENETICO3 = "genetico3"


class MetricList(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"


class ModelList(Enum):
    RESNET = "resnet"
    MOBILENET = "mobilenet"


class DatasetList(Enum):
    RPS = "RPS"
    PAINTING = "PAINTING"


class ConfiguracionGlobal:
    _instance = None  # Instancia única de la clase

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfiguracionGlobal, cls).__new__(cls)
            # Inicializa la instancia con los valores
            cls._instance.date = kwargs.get("date", "-")
            cls._instance.task_id = kwargs.get("task_id", "-")
            cls._instance.dataset_name = kwargs.get("dataset", "-")

            cls._instance.dataset = f"data/{cls._instance.dataset_name}"

        return cls._instance
