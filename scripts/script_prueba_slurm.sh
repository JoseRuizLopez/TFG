#!/bin/bash

#SBATCH --job-name prueba_jose_ruiz                 # Nombre del proceso
#SBATCH --partition dios   # Cola para ejecutar
#SBATCH --gres=gpu:1                           # Numero de gpus a usar


export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"

eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/Environments/tf2.2py36

export TFHUB_CACHE_DIR=.

python pruebatf2.py          

mail -s "Proceso finalizado" TUEMAIL@gmail.com <<< "El proceso ha finalizado"