from src.data_processing import (
    get_CIFAR_data,
    get_labels,
    get_MNIST_data,
    get_MNIST_fashion_data,
    get_simple_object,
    get_simple_object_translated,
    get_simple_pattern,
)
from src.logger import Logger
from src.loss import (
    global_mean_medians,
    highest_value,
    highest_vote,
    pixel_wise_CE,
    pixel_wise_CE_and_energy,
    pixel_wise_L2,
    pixel_wise_L2_and_CE,
    scale_loss,
)
from src.moving_nca import MovingNCA
from src.utils import get_config


def get_network(sub_path, num_data, size_img=None):
    winner_flat = Logger.load_checkpoint(sub_path)

    # Also load its config
    config = get_config(sub_path)

    # Fetch info from config and enable environment for testing
    mnist_digits = eval(config.dataset.mnist_digits)

    predicting_method = eval(config.training.predicting_method)

    moving_nca_kwargs = {
        "size_image": (config.dataset.size, config.dataset.size),
        "num_hidden": config.network.hidden_channels,
        "hidden_neurons": config.network.hidden_neurons,
        "iterations": config.network.iterations,
        "position": str(config.network.position),
        "moving": config.network.moving,
        "mnist_digits": mnist_digits,
        "img_channels": config.network.img_channels,
    }

    if size_img is not None:
        moving_nca_kwargs["size_image"] = size_img

    data_func = eval(config.dataset.data_func)
    kwargs = {
        "CLASSES": mnist_digits,
        "SAMPLES_PER_CLASS": num_data,
        "verbose": False,
        "test": True,
        "colors": True if config.network.img_channels == 3 else False,
    }

    labels = get_labels(data_func, kwargs["CLASSES"])
    # Get labels for plotting
    moving_nca_kwargs["labels"] = labels

    # Load network
    network = MovingNCA.get_instance_with(
        winner_flat, size_neo=(config.scale.test_n_neo, config.scale.test_m_neo), **moving_nca_kwargs
    )

    return network, labels, data_func, kwargs, predicting_method
