from omegaconf import DictConfig
import rtdl


def create_model(config: DictConfig, dim_in: int):
    model = rtdl.MLP.make_baseline(
        d_in=dim_in,
        d_layers=config.d_layers,
        dropout=0.1,
        d_out=1,
    )

    return model
