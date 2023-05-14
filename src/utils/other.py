import os
import numpy as np
from omegaconf import DictConfig
import rtdl
import torch


EXPERIMENTS_ROOT = 'experiments'


def apply_model(model, x_num, x_cat):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    else:
        x_all = torch.cat((x_num, x_cat), dim=1)
        return model(x_all)



def get_new_run_dir(config: DictConfig) -> str:
    experiment_dir = os.path.join('experiments', config.experiment_name)

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    
    existing_runs = [int(filename) for filename in os.listdir(experiment_dir)]

    new_run_number = np.max(existing_runs) + 1 if len(existing_runs) > 0 else 1
    new_run_name = str(new_run_number).zfill(2)

    run_dir = os.path.join(experiment_dir, new_run_name)
    os.makedirs(run_dir, exist_ok=True)

    return run_dir