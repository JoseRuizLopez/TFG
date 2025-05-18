from enum import Enum


class AlgorithmList(Enum):
    ALEATORIO = "aleatorio"
    BUSQUEDA_LOCAL = "busqueda local"
    FREE_BUSQUEDA_LOCAL = "busqueda local (libre)"
    GENETICO = "genetico"
    MEMETICO = "memetico"
    FREE_MEMETICO = "memetico (libre)"
    GENETICO2 = "genetico2"
    FREE_GENETICO2 = "genetico2 (libre)"
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


class PrintMode(Enum):
    LIBRES = "libres"
    NO_LIBRES = "no_libres"
    AMBOS = "ambos"
    JUNTOS = "juntos"


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
