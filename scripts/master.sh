#!/bin/bash

# Procesar opciones con getopt
PARSED=$(getopt -o "m:" -l MODELO: -- "$@")
if [ $? -ne 0 ]; then
    echo "Usage: $0 [-m value] [--MODELO value]"
    exit 1
fi
eval set -- "$PARSED"

# Variable para almacenar el valor de MODELO
MODELO=""
MODELO_SET=false  # Variable para saber si MODELO fue pasado como argumento

# Leer opciones
while true; do
    case "$1" in
        -m|--MODELO)
            MODELO="$2"
            MODELO_SET=true
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

# Si no se pasó -m ni --MODELO, mostrar mensaje informativo
if ! $MODELO_SET; then
    echo "No ha especificado un modelo. Puede usar '-m valor' o '--MODELO valor'."
fi

# Mostrar el valor de MODELO
echo "MODELO: ${MODELO:-'Ninguno especificado'}"

# Get current date and time, adding 0 hours (dependiendo de horario verano o invierno)
FECHA_ACTUAL=$(date -d "+0 hours" +"%Y-%m-%d_%H-%M")

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
fi

echo "Using parameters: $EXPORT_VARS"

# Submit the array job
JOBID=$(sbatch --export="$EXPORT_VARS" scripts/ejecutar_paralelo.sh | awk '{print $4}')
if [ -z "$JOBID" ]; then
    echo "Error: Failed to submit array job"
    exit 1
fi
echo "Job array submitted with ID: $JOBID"

# Submit the final job with dependency
BOXPLOT_JOB=$(sbatch --dependency=afterok:$JOBID --export="$EXPORT_VARS" scripts/generar_plots.sh | awk '{print $4}')
if [ -z "$BOXPLOT_JOB" ]; then
    echo "Error: Failed to submit boxplot job"
    exit 1
fi
echo "Boxplot job submitted with ID: $BOXPLOT_JOB"
echo "Workflow will complete after all jobs finish"
