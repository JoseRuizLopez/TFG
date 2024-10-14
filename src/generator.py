import torch
import polars as pl

from src.main import main
from utils.classes import AlgorithmList
from utils.classes import MetricList
from utils.classes import ModelList

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.is_available()}")
    porcentajes = [10, 20, 50, 100]
    evaluaciones_maximas = 100
    evaluaciones_maximas_sin_mejora = 10

    metric: MetricList = MetricList.ACCURACY
    modelo: ModelList = ModelList.RESNET
    resultados = []

    for alg in AlgorithmList:
        for ptg in porcentajes:
            result = main(
                initial_percentage=ptg,
                max_evaluations=evaluaciones_maximas,
                max_evaluations_without_improvement=evaluaciones_maximas_sin_mejora,
                algoritmo=alg.value,
                metric=metric.value,
                model_name=modelo.value
            )

            resultados.append(
                result | {
                    "Porcentaje Inicial": ptg,
                    "Algoritmo": alg.value
                }
            )

    df = pl.DataFrame(resultados, schema={
        "Algoritmo": pl.Utf8,
        "Porcentaje Inicial": pl.Int32,
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

    df.write_csv("results/resultados.csv")

    pass
