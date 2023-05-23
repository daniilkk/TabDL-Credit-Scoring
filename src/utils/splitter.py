from __future__ import annotations

import pickle
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    train: List[int]
    val: List[int]
    test: List[int]

    def __post_init__(self):
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)

        if len(train_set & val_set) > 0:
            raise RuntimeError('Train and val have non-empty intersection.')
        if len(train_set & test_set) > 0:
            raise RuntimeError('Train and test have non-empty intersection.')
        if len(val_set & test_set) > 0:
            raise RuntimeError('Val and test have non-empty intersection.')
    
    def __call__(self, part_name: str) -> List[int]:
        if not hasattr(self, part_name):
            raise RuntimeError(f'Wrong part_name: {part_name}')
        
        return getattr(self, part_name)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'train': self.train,
                'val': self.val,
                'test': self.test,
            }, f)

    @classmethod
    def load(cls, path: str) -> Split:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        split = cls(data['train'], data['val'], data['test'])

        return split


class Splitter:
    def __init__(self, all_data: pd.DataFrame):
        self.all_data = all_data
        self.split_methods = {
            'random': self._split_rows_randomly,
            'customer_id': self._split_rows_by_customer_id
        }

    @staticmethod
    def split_columns_by_feature_type(dataframe: pd.DataFrame) -> Tuple[List[str], List[str]]:
        num_features = sorted(list(dataframe.select_dtypes(include=['float64']).columns))
        cat_features = sorted(list(dataframe.select_dtypes(exclude=['float64']).columns))

        return cat_features, num_features

    def split_rows(self, method: str, train_size: float) -> Split:
        """Splits dataframe into three parts: train, val, test

        :param method: method for splitting. Must be one of [random, customer_id]
        :param train_size: size of the train part relative to the size of whole data
        :return: xs and ys for each part
        """        
        if method not in self.split_methods:
            raise RuntimeError(f'Wrong split method: "{method}". Must be one of {list(self.split_methods.keys())}')
        
        if train_size > 1 or train_size < 0:
            raise RuntimeError(f'Wrong train_size: "{train_size}". Must be in interval [0, 1]')

        split = self.split_methods[method](train_size)
        
        return split

    def _split_rows_randomly(self, train_size: float):
        indices = np.arange(len(self.all_data))
        train_indices, val_test_indices = train_test_split(indices, train_size=train_size)
        val_indices, test_indices = train_test_split(val_test_indices, train_size=0.5)

        y_all = self.all_data['Credit_Score']
        x_all = self.all_data.drop(columns='Credit_Score')

        x_all = x_all.drop(columns='Customer_ID')

        y = {
            'train': y_all.iloc[train_indices],
            'val': y_all.iloc[val_indices],
            'test': y_all.iloc[test_indices],
        }
        
        x = {
            'train': x_all.iloc[train_indices],
            'val': x_all.iloc[val_indices],
            'test': x_all.iloc[test_indices]
        }

        return x, y

    def _split_rows_by_customer_id(self, train_size: float) -> Split:
        train_customer_ids, val_test_customer_ids = train_test_split(
            self.all_data['Customer_ID'].unique(),
            train_size=train_size
        )
        val_customer_ids, test_customer_ids = train_test_split(val_test_customer_ids, train_size=0.5)

        train_ids = self.all_data.index[self.all_data['Customer_ID'].isin(set(train_customer_ids))].tolist()
        val_ids = self.all_data.index[self.all_data['Customer_ID'].isin(set(val_customer_ids))].tolist()
        test_ids = self.all_data.index[self.all_data['Customer_ID'].isin(set(test_customer_ids))].tolist()

        return Split(train_ids, val_ids, test_ids)

        # train = self.all_data[self.all_data['Customer_ID'].isin(set(train_customer_ids))]
        # val = self.all_data[self.all_data['Customer_ID'].isin(set(val_customer_ids))]
        # test = self.all_data[self.all_data['Customer_ID'].isin(set(test_customer_ids))]

        # train = train.drop(columns='Customer_ID')
        # test = test.drop(columns='Customer_ID')
        # val = val.drop(columns='Customer_ID')

        # y = {
        #     'train': train['Credit_Score'],
        #     'val': val['Credit_Score'],
        #     'test': test['Credit_Score'],
        # }

        # x = {
        #     'train': train.drop(columns='Credit_Score'),
        #     'val': val.drop(columns='Credit_Score'),
        #     'test': test.drop(columns='Credit_Score'),
        # }

        # return x, y
