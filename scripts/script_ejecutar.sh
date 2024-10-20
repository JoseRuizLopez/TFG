#!/bin/bash

#SBATCH --job-name entrenamiento_jose_ruiz                 # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar


export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export TORCH_HOME="/mnt/homeGPU/joruiz/TFG/.cache/torch"
export MPLCONFIGDIR="/mnt/homeGPU/joruiz/TFG/.cache/matplotlib"
export PYTHONPATH="$PYTHONPATH:/mnt/homeGPU/joruiz/TFG"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export SERVER="gpu"


eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/joruiz/TFG/pt2.3py3.10

export TFHUB_CACHE_DIR=.

# Obtener la fecha y hora actual en formato YYYY-MM-DD_HH-MM
fecha_actual=$(date -d "+2 hours" +"%Y-%m-%d_%H-%M")

# Definir el nombre del archivo de salida con la fecha y hora
archivo_salida="results/salida_${fecha_actual}.txt"

# Ejecutar el script de Python y redirigir la salida al archivo
python src/generator.py > "$archivo_salida"

mail -s "Proceso finalizado" ruizlopezjose@correo.ugr.es <<< "El proceso ha finalizado"
