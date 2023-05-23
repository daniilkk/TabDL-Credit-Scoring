from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from src.ple import PiecewiseLinearEncoder

from src.utils.splitter import Split, Splitter


class TabDataModule:
    def __init__(
            self,
            data_path: str,
            train_size: float,
            split_method: str,
            device: torch.device,
            cat_encoder: str,
            num_encoder: Optional[str] = None,
            split: Optional[Split] = None
    ):  
        self.data_path = data_path
        self.train_size = train_size
        self.split_method = split_method
        self.num_encoder: Optional[PiecewiseLinearEncoder] = num_encoder
        self.cat_encoder = cat_encoder
        self.device = device

        self.label_encoder: Optional[LabelEncoder] = None
        self.data_scaler: Optional[StandardScaler] = None
        self.cat_cardinalities = None
        self.split = split
        self.parts_names = ('train', 'val', 'test')
        
        self.datasets = self._prepare_datasets(self.data_path)
    
    def get_dataloader(self, part_name: str, batch_size: int, shuffle: bool):
        if part_name not in self.parts_names:
            raise RuntimeError(f'Wrong part_name: {part_name}')

        return DataLoader(self.datasets[part_name], batch_size=batch_size, shuffle=shuffle, drop_last=True)

    def _prepare_datasets(self, data_path: str):
        x_parts, y_parts = self._load_and_prepare_data(data_path)

        datasets = {
            part_name: TabDataset(x_parts[part_name], y_parts[part_name])
            for part_name in self.parts_names
        }
        
        return datasets

    def _load_and_prepare_data(self, data_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        data = pd.read_csv(data_path)

        if self.split is None:
            self.split = Splitter(data).split_rows(self.split_method, self.train_size)

        data = data.drop(columns='Customer_ID')

        x = data.drop(columns='Credit_Score')
        y = data['Credit_Score']

        cat_features, num_features = Splitter.split_columns_by_feature_type(x)
        x = self._split_features_by_type(x, cat_features, num_features)

        y = self._encode_labels(y)
        x = self._scale_num_features(x)
        x = self._encode_num_features(x)
        x = self._encode_cat_features(x)

        feature_type_to_dtype = {
            'cat': torch.int64 if self.cat_encoder == 'ordinal' else torch.float32,
            'num': torch.float32
        }
        
        x_parts = {part_name: {} for part_name in self.parts_names}
        y_parts = {}
        for part_name in self.parts_names:
            for feature_type in ('cat', 'num'):
                x_parts[part_name][feature_type] = torch.tensor(
                    x[feature_type].iloc[self.split(part_name)].values,
                    device=self.device,
                    dtype=feature_type_to_dtype[feature_type]
                )

            y_parts[part_name] = torch.tensor(
                y[self.split(part_name)],
                device=self.device,
                dtype=torch.float32
            )
        
        return x_parts, y_parts

    def _split_features_by_type(
            self,
            x: pd.DataFrame,
            cat_features: List[str],
            num_features: List[str],
        ):
        x_splitted = {
            'cat': x[cat_features],
            'num': x[num_features],
        } 

        return x_splitted

    def _encode_labels(self, y: pd.Series):
        self.label_encoder = LabelEncoder().fit(y.iloc[self.split.train])
        y = self.label_encoder.transform(y)

        return y

    def _scale_num_features(self, x: Dict[str, pd.DataFrame]):
        self.data_scaler = StandardScaler().fit(x['num'].iloc[self.split.train])
        x['num'] = pd.DataFrame(
            data=self.data_scaler.transform(x['num']),
            columns=x['num'].columns
        )

        return x

    def _encode_num_features(self, x: Dict[str, pd.DataFrame]):
        if self.num_encoder is None:
            return x

        self.num_encoder.fit(x['num'].iloc[self.split.train])
        x['num'] = self.num_encoder.transform(x['num'])
        
        return x

    def _encode_cat_features(self, x: Dict[str, pd.DataFrame]):
        cat_features = list(x['cat'].columns)

        if self.cat_encoder == 'ohe':
            column_transformer = make_column_transformer(
                (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features),
                remainder='passthrough'
            )
        elif self.cat_encoder == 'ordinal':
            column_transformer = make_column_transformer(
                (
                    OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    ), 
                    cat_features
                ),
                remainder='passthrough'
            )
        else:
            raise RuntimeError(f'Wrong cat_encoder: {self.cat_encoder}')

        column_transformer = column_transformer.fit(x['cat'].iloc[self.split.train])
        if self.cat_encoder == 'ordinal':
            self.cat_cardinalities = [
                len(feature_categories) + 1
                for feature_categories in column_transformer.transformers_[0][1].categories_
            ]

        x['cat'] = pd.DataFrame(
            column_transformer.transform(x['cat']),
            columns=column_transformer.get_feature_names_out()
        )

        return x


class TabDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.labels = labels
        self.data = data
        self.n_features_num = self.data['num'].shape[1]
        self.n_features_all = self.n_features_num + self.data['cat'].shape[1]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        x_item = (self.data['num'][index], self.data['cat'][index])
        y_item = self.labels[index]
        
        return x_item, y_item 
    