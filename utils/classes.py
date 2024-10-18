from enum import Enum


class AlgorithmList(Enum):
    ALEATORIO = "aleatorio"
    BUSQUEDA_LOCAL = "busqueda local"
    GENETICO = "genetico"
    # MEMETICO = "memetico"


class MetricList(Enum):
    ACCURACY = "accuracy"
    F1 = "f1"


class ModelList(Enum):
    # RESNET = "resnet"
    MOBILENET = "mobilenet"
