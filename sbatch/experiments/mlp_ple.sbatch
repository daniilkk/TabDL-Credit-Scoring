#!/bin/bash

#SBATCH --job-name=mlp_ple           # Название задачи
#SBATCH --error=output/mlp_ple-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/mlp_ple.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU
#SBATCH --time=200

module purge
module load Python

deactivate
source activate venv

pip install -r requirements.txt

export PYTHONPATH=$(pwd)

for _ in $(seq 1 5); do
    python -u src/train.py --experiment mlp_ple
done
