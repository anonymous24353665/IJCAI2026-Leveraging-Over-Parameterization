import logging
from typing import Any, List, Tuple
from torch import nn, Tensor

from Generators.ArchitecturesGenerator.one_rs_param.hyper_params_search import BinaryHyperParamsResearch
from Generators.ArchitecturesGenerator.utils.nn_models import CustomConvNN
from Generators.ArchitecturesGenerator.one_rs_param.regularized_trainer import ModelTrainingManager
from Generators.ArchitecturesGenerator.one_rs_param.config import load_config
from Generators.ArchitecturesGenerator.utils.rs_loss_regularizer import calculate_rs_loss_regularizer_conv
from Generators.ArchitecturesGenerator.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Modulo train_conv caricato")

# =========================
# Custom ModelTrainingManager per rete convoluzionale
# =========================
class ModelTrainingManagerConv(ModelTrainingManager):
    def get_rsloss(
        self,
        model: nn.Module,
        model_ref,
        architecture_tuple: Tuple,
        input_batch: Tensor,
        perturbation,
        eps: float,
        method: str = "ibp"
    ) -> Tuple[Any, Any]:
        """
        Calcola il Regularization Sensitivity Loss per reti convoluzionali.
        """
        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_conv(
            model,
            architecture_tuple,
            input_batch,
            eps,
            method=method,
            normalized=True
        )
        return rs_loss, n_unstable_nodes


# =========================
# Funzione principale run_training
# =========================
def run_training(
    dataset_name: str,
    hidden_layers_dim: List[int],
    config_path_ini,
    config_path_yaml,
    input_dim: int = 28,
    output_dim: int = 10,
    conv_filters_dim: int = 17,
    min_increment: float = 0.1,
    max_increment: float = 9,
    steps_limit: int = 10,
    rs_loss_bool: bool = True,
    skip_binary_search: bool = False,
    debug=False,
):
    """
    Esegue la ricerca dei migliori iperparametri per una rete convoluzionale shallow.
    """
    config = load_config(config_path_ini)

    # Impostazioni dataset-specific
    stride = 1
    padding = 0

    if dataset_name == "MNIST":
        kernel_size = 5
    elif dataset_name == "FMNIST":
        kernel_size = 2
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")

    # Costruzione tuple architettura: [(input_dim, output_dim, conv_filters, kernel, stride, padding, fc_dim)]
    arch_tuple = [
        (input_dim, output_dim, conv_filters_dim, kernel_size, stride, padding, hidden_layers_dim[i])
        for i in range(len(hidden_layers_dim))
    ]

    # Inizializza il motore di ricerca iperparametri
    hyper_params_search = BinaryHyperParamsResearch(
        CustomConvNN,
        config_path_yaml,
        config,
        dataset_name,
        arch_tuple,
        rs_loss_bool=rs_loss_bool,
        skip_binary_search=skip_binary_search
    )

    # Esegui la ricerca binaria
    hyper_params_search.binary_search(
        min_increment,
        max_increment,
        steps_limit,
        ModelTrainingManagerConv
    )
