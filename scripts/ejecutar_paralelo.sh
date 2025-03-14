#!/bin/bash

#SBATCH --job-name entrenamiento_jose_ruiz                 # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                                       # Numero de gpus a usar
#SBATCH --array=0-4                                        # Rango para tareas paralelas
#SBATCH --exclude=atenea                                   # Excluir nodo "atenea"

# Verify FECHA_ACTUAL is set
if [ -z "$FECHA_ACTUAL" ]; then
    echo "Error: FECHA_ACTUAL variable is not set"
    exit 1
fi

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

# Extract the date directory structure from FECHA_ACTUAL
DATE_DIR=$(dirname "results/salidas/${FECHA_ACTUAL}")

# Create the directory structure
mkdir -p "$DATE_DIR"

# Utiliza la variable CURRENT_TIME que se recibió del master_script
archivo_salida="results/salidas/${FECHA_ACTUAL}_task_${SLURM_ARRAY_TASK_ID}.txt"
echo "Executing task ${SLURM_ARRAY_TASK_ID} with output to ${archivo_salida}"

pwd

# Construct command line parameters
PYTHON_ARGS=()
PYTHON_ARGS+=("--task_id" "$SLURM_ARRAY_TASK_ID")
PYTHON_ARGS+=("--FECHA_ACTUAL" "$FECHA_ACTUAL")

# Only add MODELO if it's defined
if [ -n "$MODELO" ]; then
    PYTHON_ARGS+=("--MODELO" "$MODELO")
fi

python src/generator.py "${PYTHON_ARGS[@]}" > "$archivo_salida"

