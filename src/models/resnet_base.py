from omegaconf import DictConfig
import rtdl


def create_model(config: DictConfig, dim_in: int):
    model = rtdl.ResNet.make_baseline(
        d_in=dim_in,
        d_main=config.d_main,
        d_hidden=config.d_hidden,
        dropout_first=config.dropout_first,
        dropout_second=config.dropout_second,
        n_blocks=config.n_blocks,
        d_out=1,
    )

    return model


if __name__ == '__main__':
    model = create_model(None, 10)
    print(model)
