

import numpy as np
import pandas as pd


class PiecewiseLinearEncoder:
    def __init__(self, n_bins: int):
        self.n_bins = n_bins

        self.feature_name_to_borders = {}

    def fit(self, x: pd.DataFrame):
        for feature_name in x.columns:
            values = x[feature_name].values
            min_value, max_value = values.min(), values.max()

            inner_borders = np.quantile(values, np.arange(1, self.n_bins) / self.n_bins)
            borders = [min_value, *inner_borders, max_value]

            # print(borders)

            self.feature_name_to_borders[feature_name] = borders

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        features_encoded = []

        for feature_name in x.columns:
            values = x[feature_name].values
            feature_encoded = np.zeros((len(values), self.n_bins))

            borders = self.feature_name_to_borders[feature_name]

            digits = np.digitize(values, borders[1: -1])

            for idx, value in enumerate(values):
                value = min(max(borders[0], value), borders[-1])

                feature_encoded[idx, digits[idx] + 1:] = 1

                left_border = borders[digits[idx]]
                right_border = borders[digits[idx] + 1]

                feature_encoded[idx, digits[idx]] = (value - left_border) / (right_border - left_border)

            features_encoded.append(feature_encoded)

        return pd.DataFrame(np.concatenate(features_encoded, axis=1))


if __name__ == '__main__':
    ple = PiecewiseLinearEncoder(3)

    train = pd.DataFrame.from_dict({
        'col1': [0, 2, 5, 10, 7],
        'col2': [0, 1, 6, 5, 10],
    })

    test = pd.DataFrame.from_dict({
        'col1': [-1, 11, 4, 6, 1e10],
        'col2': [-10, 1, -6, 5, 10],
    })
    
    ple.fit(train)

    print(ple.transform(train))
    print(ple.transform(test))