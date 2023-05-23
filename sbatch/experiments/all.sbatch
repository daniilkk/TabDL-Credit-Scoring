#!/bin/bash

#SBATCH --job-name=all           # Название задачи
#SBATCH --error=output/all-%j.err        # Файл для вывода ошибок
#SBATCH --output=output/all.log       # Файл для вывода результатов

#SBATCH --ntasks=1                # Количество MPI процессов
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU
#SBATCH --time=1500

echo $SHELL

module purge
module load Python

deactivate
conda activate venv3

pip install -r requirements.txt

export PYTHONPATH=$(pwd)

for exp_name in catboost mlp mlp_ple resnet resnet_ple ftt ftt_ple; do
    for _ in $(seq 1 5); do
        python -u src/train.py --experiment $exp_name
    done
done
