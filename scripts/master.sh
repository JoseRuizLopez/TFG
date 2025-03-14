#!/bin/bash

# Process options with getopt
PARSED=$(getopt -o "" -l MODELO: -- "$@")
if [ $? -ne 0 ]; then
    echo "Usage: $0 [--MODELO value]"
    exit 1
fi
eval set -- "$PARSED"

MODELO=""

while true; do
  case "$1" in
    --MODELO) MODELO="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Usage: $0 [--MODELO value]"; exit 1 ;;
  esac
done

# Get current date and time, adding 1 hours
FECHA_ACTUAL=$(date -d "+1 hours" +"%Y-%m-%d_%H-%M")
echo "Using timestamp: $FECHA_ACTUAL"


# Format date for output filename - create a safe copy of the variable
FECHA_FORMATEADA=$FECHA_ACTUAL

# Reemplazar los dos primeros '-'
FECHA_FORMATEADA=${FECHA_FORMATEADA/-/\/}  # Primer '-'
FECHA_FORMATEADA=${FECHA_FORMATEADA/-/\/}  # Segundo '-'

# Reemplazar el primer '_'
FECHA_FORMATEADA=${FECHA_FORMATEADA/_/\/}

# Create export variables string for sbatch
EXPORT_VARS="ALL,FECHA_ACTUAL=$FECHA_FORMATEADA"
if [ -n "$MODELO" ]; then
    EXPORT_VARS="$EXPORT_VARS,MODELO=$MODELO"
    echo "Using model: $MODELO"
fi

# Submit the array job
JOBID=$(sbatch --export="$EXPORT_VARS" scripts/ejecutar_paralelo.sh | awk '{print $4}')
if [ -z "$JOBID" ]; then
    echo "Error: Failed to submit array job"
    exit 1
fi
echo "Job array submitted with ID: $JOBID"

# Submit the final job with dependency
BOXPLOT_JOB=$(sbatch --dependency=afterok:$JOBID --export="$EXPORT_VARS" scripts/generar_boxplot.sh | awk '{print $4}')
if [ -z "$BOXPLOT_JOB" ]; then
    echo "Error: Failed to submit boxplot job"
    exit 1
fi
echo "Boxplot job submitted with ID: $BOXPLOT_JOB"
echo "Workflow will complete after all jobs finish"
