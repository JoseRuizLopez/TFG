#!/bin/bash

#SBATCH --job-name entrenamiento_jose_ruiz                 # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar
#SBATCH --exclude=atenea                                   # Excluir nodo "atenea"

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

# Get current date and time, adding 1 hours
FECHA_ACTUAL=$(date -d "+1 hours" +"%Y-%m-%d_%H-%M")

# Format date for output filename - create a safe copy of the variable
FECHA_FORMATEADA=$FECHA_ACTUAL

# Reemplazar los dos primeros '-'
FECHA_FORMATEADA=${FECHA_FORMATEADA/-/\/}  # Primer '-'
FECHA_FORMATEADA=${FECHA_FORMATEADA/-/\/}  # Segundo '-'

# Reemplazar el primer '_'
FECHA_FORMATEADA=${FECHA_FORMATEADA/_/\/}

# Extract the date directory structure from FECHA_ACTUAL
DATE_DIR=$(dirname "results/salidas/${FECHA_FORMATEADA}")

# Definir el nombre del archivo de salida con la fecha y hora
archivo_salida="results/salidas/salida_${fecha_actual}.txt"

# Ejecutar el script de Python y redirigir la salida al archivo
python src/generator.py --task_id "-1" > "$archivo_salida"

mail -s "Proceso finalizado" ruizlopezjose@correo.ugr.es <<< "El proceso ha finalizado"
