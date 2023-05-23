import os
import pickle
from typing import Any, Dict, List

import numpy as np
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import TabDataModule
from src.ple import PiecewiseLinearEncoder
from src.utils.metrics import compute_metrics
from src.utils.other import DATA_ROOT, EXPERIMENTS_ROOT, apply_model, get_model_creator, load_experiment_config
from src.utils.splitter import Split


def load_best_checkpoint(run_dir_path: str) -> Dict[str, Any]:
    train_val_metrics_path = os.path.join(run_dir_path, 'train_val_metrics.csv')
    metrics_df = pd.read_csv(train_val_metrics_path)
    best_epoch = int(metrics_df.iloc[metrics_df['val_auprc'].idxmax()]['epoch'])

    checkpoint_path = os.path.join(run_dir_path, 'checkpoints', f'{str(best_epoch).zfill(4)}.pt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    return checkpoint


@torch.no_grad()
def inference_model_on_test(
        model: nn.Module,
        checkpoint: Dict[str, Any],
        data_module: TabDataModule,
        batch_size: int
) -> Dict[str, float]:
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataloader = data_module.get_dataloader('test', batch_size, shuffle=False)

    predict = []
    target = []
    for batch in dataloader:
        x_batch, y_batch = batch

        predict.append(apply_model(model, *x_batch))
        target.append(y_batch)

    predict = torch.cat(predict).squeeze(1).cpu().numpy()
    predict = np.round(sp.special.expit(predict))

    target = torch.cat(target).cpu().numpy()

    metrics = compute_metrics(predict, target)

    return metrics


def save_experiment_metrics(experiment_metrics: List[dict], save_path: str):
    metrics_df = pd.DataFrame.from_records(experiment_metrics)

    metrics_df = pd.concat(
        (metrics_df, pd.DataFrame.from_dict({'run': ['avg'], 'auprc': [metrics_df['auprc'].mean()]})),
        ignore_index=True
    )
    
    metrics_df.to_csv(save_path, index=False)


def main(exclude_experiments: List[str]):
    TEST_METRICS_FILENAME = 'test_metrics.csv'

    for experiment_name in os.listdir(EXPERIMENTS_ROOT):
        if experiment_name in exclude_experiments:
            continue
        
        config = load_experiment_config(experiment_name)
        experiment_dir_path = os.path.join(EXPERIMENTS_ROOT, experiment_name)

        experiment_metrics = []

        experiment_runs = list(sorted([
            filename for filename in os.listdir(experiment_dir_path)
            if os.path.isdir(os.path.join(experiment_dir_path, filename))
        ]))
        
        for run_name in experiment_runs:
            print(experiment_name, run_name)
            run_dir_path = os.path.join(experiment_dir_path, run_name)

            split_path = os.path.join(run_dir_path, 'datamodule.pickle')
            split = Split.load(split_path)

            data_path = os.path.join(DATA_ROOT, f'{config.data}.csv')
            data_module = TabDataModule(
                data_path=data_path,
                train_size=config.train_size,
                split_method=config.split_method,
                device=torch.device('cpu'),
                num_encoder= PiecewiseLinearEncoder(config.ple_n_bins) if config.use_ple else None,
                cat_encoder= 'ordinal' if config.model_type == 'ft_transformer' else 'ohe',
                split=split
            )

            create_model_kwargs = {
                'n_num_features': data_module.datasets['train'].n_features_num,
                'cat_cardinalities': data_module.cat_cardinalities
            } if config.model_type == 'ft_transformer' else {
                'dim_in': data_module.datasets['train'].n_features_all
            }
            model = get_model_creator(config.model)(config, **create_model_kwargs)
            
            checkpoint = load_best_checkpoint(run_dir_path)

            run_metrics = inference_model_on_test(
                model,
                checkpoint,
                data_module,
                config.batch_size
            )

            experiment_metrics.append({'run': run_name, 'auprc': run_metrics['auprc']})

        save_experiment_metrics(experiment_metrics, os.path.join(experiment_dir_path, TEST_METRICS_FILENAME))


if __name__ == '__main__':
    EXCLUDE_EXPERIMENTS = ['catboost']

    main(EXCLUDE_EXPERIMENTS)
