from main import get_from_config
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
)
from src.moving_nca_no_tf import MovingNCA
from src.utils import get_config


def get_network(sub_path, num_data, size_img=None):
    winner_flat = Logger.load_checkpoint(sub_path)

    # Also load its config
    config = get_config(sub_path)
    moving_nca_kwargs, loss_function, predicting_method, data_func, kwargs = get_from_config(config)

    # Allow image size to be adjusted
    if size_img is not None:
        moving_nca_kwargs["size_image"] = size_img

    # Alter data kwargs
    kwargs["test"] = True
    kwargs["SAMPLES_PER_CLASS"] = num_data

    # Get labels
    labels = moving_nca_kwargs["labels"]

    # Load network
    network = MovingNCA.get_instance_with(
        winner_flat, size_neo=(config.scale.train_n_neo, config.scale.train_m_neo), **moving_nca_kwargs
    )

    return network, labels, data_func, kwargs, predicting_method
