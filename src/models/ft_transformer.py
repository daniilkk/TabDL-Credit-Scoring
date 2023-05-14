from typing import List
from omegaconf import DictConfig
import rtdl


def create_model(config: DictConfig, n_num_features: int, cat_cardinalities: List[int]):
    model = rtdl.FTTransformer.make_default(
    n_num_features=n_num_features,
    cat_cardinalities=cat_cardinalities,
    last_layer_query_idx=[-1],
    d_out=1
)

    return model


if __name__ == '__main__':
    model = create_model(None, 10)
    print(model)
