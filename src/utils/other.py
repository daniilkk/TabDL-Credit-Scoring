from importlib import import_module
import os
from typing import Callable

import numpy as np
from omegaconf import DictConfig, OmegaConf
import rtdl
import torch
import torch.nn as nn


EXPERIMENTS_ROOT = 'experiments/'
DATA_ROOT = 'data/csv/'
MODELS_ROOT = 'src/models/'
CONFIGS_ROOT = 'configs/'


def apply_model(model, x_num, x_cat):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    else:
        x_all = torch.cat((x_num, x_cat), dim=1)
        return model(x_all)


def load_experiment_config(experiment_name: str) -> DictConfig:
    config_path = os.path.join(CONFIGS_ROOT, f'{experiment_name}.yaml')

    config = OmegaConf.load(config_path)

    return config


def get_model_creator(model_name: str) -> Callable[[], nn.Module]:
    model_module_path = os.path.join(MODELS_ROOT, model_name).replace('/', '.')
    model_module = import_module(model_module_path)

    if not hasattr(model_module, 'create_model'):
        raise RuntimeError(f'Module {model_module_path} does not have an attribute create_model')

    return model_module.create_model


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