#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gpus=1

module purge
module load Python

source deactivate
source activate my_py_env

nvidia-smi	
which python
python -V
python -c 'import torch; print(torch.cuda.is_available())'
