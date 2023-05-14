#!/bin/bash

#SBATCH --job-name=$2           # Название задачи
#SBATCH --error=output/$1-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/$1-%j.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU
#SBATCH --time=30

module purge
module load Python

source deactivate
source activate venv

export PYTHONPATH=$(pwd)

for _ in $(seq 1 $3); do
    python src/run.py --config configs/$2.yaml
done
