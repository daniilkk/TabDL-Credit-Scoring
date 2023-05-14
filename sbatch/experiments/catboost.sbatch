#!/bin/bash

#SBATCH --job-name=catboost           # Название задачи
#SBATCH --error=output/catboost-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/catboost.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU
#SBATCH --time=200

module purge
module load Python

deactivate
conda activate venv

export PYTHONPATH=$(pwd)

for _ in $(seq 1 5); do
    python -u src/run.py --config configs/catboost.yaml
done
