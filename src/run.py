import argparse
import os
import csv
import shutil
from importlib import import_module
from typing import Dict, List, Tuple
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import rtdl

from src.dataset import TabDataModule
from src.ple import PiecewiseLinearEncoder
from src.utils.metrics import compute_metrics, dump_metrics
from src.utils.other import get_new_run_dir
from src.utils.splitter import Splitter
from src.trainer import Trainer


DATA_ROOT = 'data/csv/'
MODELS_ROOT = 'src/models'


def train_catboost(config: DictConfig):
    data_path = os.path.join(DATA_ROOT, f'{config.data}.csv')
    data = pd.read_csv(data_path)
    
    x, y = Splitter(data).split_rows(config.split_method, config.train_size)

    label_encoder = LabelEncoder()
    y['train'] = label_encoder.fit_transform(y['train'])
    y['test'] = label_encoder.transform(y['test'])

    model = CatBoostClassifier(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
    )

    cat_features, num_features = Splitter.split_columns_by_feature_type(x['train'])

    model.fit(x['train'], y['train'], cat_features=cat_features, verbose=10)

    predictions_proba = model.predict_proba(x['test'])[:, 1]
    metrics = compute_metrics(predictions_proba, y['test'])

    run_dir = get_new_run_dir(config)

    dump_metrics(metrics, os.path.join(run_dir, 'test_metrics.csv'))


def train_neural_model(config: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_dir = get_new_run_dir(config)

    data_path = os.path.join(DATA_ROOT, f'{config.data}.csv')
    data_module = TabDataModule(
        data_path=data_path,
        train_size=config.train_size,
        split_method=config.split_method,
        device=device,
        num_encoder= PiecewiseLinearEncoder(config.ple_n_bins) if config.use_ple else None,
        cat_encoder= 'ordinal' if config.model_type == 'ft_transformer' else 'ohe',
    )

    model_module = import_module(os.path.join(MODELS_ROOT, config.model).replace('/', '.'))

    create_model_kwargs = {
        'n_num_features': data_module.datasets['train'].n_features_num,
        'cat_cardinalities': data_module.cat_cardinalities
    } if config.model_type == 'ft_transformer' else {
        'dim_in': data_module.datasets['train'].n_features_all
    }

    model = model_module.create_model(
        config,
        **create_model_kwargs
    ).to(device)

    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay)
    )

    loss_fn = F.binary_cross_entropy_with_logits

    trainer = Trainer(model, data_module, optimizer, loss_fn, run_dir)
    trainer.train(
        n_epochs=1000,
        batch_size=config.batch_size,
        report_frequency=1000
    )


def main(config: DictConfig):
    if config.model_type == 'catboost':
        train_catboost(config)
    else:
        train_neural_model(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config')
    
    config = OmegaConf.load(parser.parse_args().config)

    main(config)
