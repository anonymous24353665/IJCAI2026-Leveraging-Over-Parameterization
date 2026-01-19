import cProfile
import logging
import os
import pstats
from typing import Any, List, Tuple

import torch
from torch import nn, Tensor

from Generators.ArchitecturesGenerator.one_rs_param.hyper_params_search import BinaryHyperParamsResearch
from Generators.ArchitecturesGenerator.utils.nn_models import CustomFCNN_Shallow
from Generators.ArchitecturesGenerator.one_rs_param.regularized_trainer import ModelTrainingManager
from Generators.ArchitecturesGenerator.one_rs_param.config import load_config
from Generators.ArchitecturesGenerator.utils.rs_loss_regularizer import calculate_rs_loss_regularizer_fc
from Generators.ArchitecturesGenerator.utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)
logger.info("Modulo train_shallow caricato")


# Custom ModelTrainingManager per rete shallow
class ModelTrainingManagerShallow(ModelTrainingManager):
    def get_rsloss(
        self,
        model: nn.Module,
        model_ref,
        architecture_tuple: Tuple,
        input_batch: Tensor,
        perturbation,
        eps,
        method: str = "ibp"
    ) -> Tuple[Any, Any]:
        input_batch = input_batch[0]
        rs_loss, n_unstable_nodes = calculate_rs_loss_regularizer_fc(
            model_ref,
            architecture_tuple[1],
            input_batch,
            eps,
            normalized=True
        )
        return rs_loss, n_unstable_nodes


def run_training(
    dataset_name: str,
    config_path_ini: str,
    config_path_yaml: str,
    hidden_layers_dim: List[int],
    min_increment: float = 0.1,
    max_increment: float = 6,
    steps_limit: int = 3,
    rs_loss_bool: bool = True,
    skip_binary_search: bool = False,
    debug=False,
):
    # Carica configurazione
    config = load_config(config_path_ini)

    # Costruzione tuple architettura: [(784, h, 10), ...]
    architectures = [(784, x, 10) for x in hidden_layers_dim]


    hyper_params_search = BinaryHyperParamsResearch(
            CustomFCNN_Shallow,
            config_path_yaml,
            config,
            dataset_name,
            architectures,
            rs_loss_bool=rs_loss_bool,
            skip_binary_search=skip_binary_search
        )

    if debug:
        profiler = cProfile.Profile()
        profiler.enable()

    hyper_params_search.binary_search(
        min_increment,
        max_increment,
        steps_limit,
        ModelTrainingManagerShallow,
    )

    if debug:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats()
