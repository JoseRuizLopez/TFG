#!/bin/bash

#SBATCH --job-name entrenamiento_jose_ruiz                 # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar

0
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/joruiz/TFG/pt2.3py3.10

export TFHUB_CACHE_DIR=.

python src/dataset_loader.py > salida.txt

mail -s "Proceso finalizado" ruizlopezjose@correo.ugr.es <<< "El proceso ha finalizado"