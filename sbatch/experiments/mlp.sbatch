#!/bin/bash

#SBATCH --job-name=mlp           # Название задачи
#SBATCH --error=output/mlp-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/mlp-%j.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU
#SBATCH --time=100

module purge
module load Python

source deactivate
source activate venv

export PYTHONPATH=$(pwd)

for _ in $(seq 1 4); do
    python -u src/run.py --config configs/mlp.yaml
done
