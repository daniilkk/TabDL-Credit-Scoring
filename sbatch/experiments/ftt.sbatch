#!/bin/bash

#SBATCH --job-name=ftt           # Название задачи
#SBATCH --error=output/ftt-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/ftt.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=4          # Требуемое кол-во CPU
#SBATCH --time=300

module purge
module load Python

conda init bash 
source /home/dkrasilnikov/.bashrc

deactivate
source activate venv

pip install -r requirements.txt

export PYTHONPATH=$(pwd)

for _ in $(seq 1 5); do
    python -u src/train.py --experiment ftt
done
