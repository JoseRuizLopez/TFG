from enum import Enum


class AlgorithmList(Enum):
    BUSQUEDA_LOCAL = "busqueda local"
    ALEATORIO = "aleatorio"
    GENETICO = "genetico"
    # MEMETICO = "memetico"


class MetricList(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"


class ModelList(Enum):
    RESNET = "resnet"
    MOBILENET = "mobilenet"
