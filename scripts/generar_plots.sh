#!/bin/bash
#SBATCH --job-name=boxplot
#SBATCH --partition=dios
#SBATCH --gres=gpu:1
#SBATCH --exclude=atenea                                   # Excluir nodo "atenea"

# Process command line arguments if script is called directly
# Otherwise, use environment variables passed by sbatch
if [ "$#" -gt 0 ]; then
    # Parse command-line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --FECHA_ACTUAL)
                FECHA_ACTUAL="$2"
                shift 2
                ;;
            --MODELO)
                MODELO="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                exit 1
                ;;
        esac
    done
fi

# Verify FECHA_ACTUAL is set (either from command line or environment)
if [ -z "$FECHA_ACTUAL" ]; then
    echo "Error: FECHA_ACTUAL variable is not set"
    echo "Usage: $0 --FECHA_ACTUAL <fecha> [--MODELO <modelo>]"
    exit 1
fi

# Configuración del entorno
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
export TORCH_HOME="/mnt/homeGPU/joruiz/TFG/.cache/torch"
export MPLCONFIGDIR="/mnt/homeGPU/joruiz/TFG/.cache/matplotlib"
export PYTHONPATH="$PYTHONPATH:/mnt/homeGPU/joruiz/TFG"
export CUBLAS_WORKSPACE_CONFIG=:16:8
export SERVER="gpu"

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/joruiz/TFG/pt2.3py3.10

export TFHUB_CACHE_DIR=.

# Construct command line parameters
PYTHON_ARGS=()
PYTHON_ARGS+=("--FECHA_ACTUAL" "$FECHA_ACTUAL")

# Only add MODELO if it's defined
if [ -n "$MODELO" ]; then
    PYTHON_ARGS+=("--MODELO" "$MODELO")
fi

echo "Running boxplot generation with parameters: ${PYTHON_ARGS[@]}"

# Execute the Python script with the parameters
python src/generar_plots.py "${PYTHON_ARGS[@]}"
