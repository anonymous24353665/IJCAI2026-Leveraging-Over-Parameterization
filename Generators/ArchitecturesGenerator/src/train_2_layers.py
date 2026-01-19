import logging
import torch
import cProfile
import pstats
from typing import Any, Tuple, List
from torch import nn

from Generators.ArchitecturesGenerator.one_rs_param.hyper_params_search import (
    BinaryHyperParamsResearch,
)
from Generators.ArchitecturesGenerator.utils.nn_models import CustomFCNN
from Generators.ArchitecturesGenerator.one_rs_param.regularized_trainer import (
    ModelTrainingManager,
)
from Generators.ArchitecturesGenerator.one_rs_param.config import load_config
from Generators.ArchitecturesGenerator.utils.rs_loss_regularizer import (
    calculate_rs_loss_regularizer_fc_2_layers,
)
from Generators.ArchitecturesGenerator.utils.logger import setup_logger


def run_training(
    dataset_name: str,
    hidden_layers_dim: List[int],
    config_path_ini: str,
    config_path_yaml: str,
    min_increment: float = 0.1,
    max_increment: float = 6,
    steps_limit: int = 3,
    rs_loss_bool: bool = True,
    skip_binary_search: bool = False,
    debug: bool = False
):

    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Applicazione avviata")

    class ModelTrainingManagerComplex(ModelTrainingManager):
        def get_rsloss(
            self,
            model: nn.Module,
            model_ref,
            architecture_tuple: tuple,
            input_batch: Tuple,
            perturbation,
            eps,
            method: str = "ibp",
        ) -> tuple[Any, Any]:

            input_lb = torch.clamp(input_batch[0] - eps, min=0, max=1)
            input_ub = torch.clamp(input_batch[0] + eps, min=0, max=1)

            rs_loss, n_unstable_nodes = (
                calculate_rs_loss_regularizer_fc_2_layers(
                    model_ref,
                    architecture_tuple[2],
                    input_lb,
                    input_ub,
                    normalized=True,
                )
            )

            return rs_loss, n_unstable_nodes

    # Load config
    config = load_config(config_path_ini)


    # Build architecture tuples: (784, (2, h), 10)
    hidden_layer_tuples = [(2, dim) for dim in hidden_layers_dim]
    architectures = [(784, x, 10) for x in hidden_layer_tuples]

    hyper_params_search = BinaryHyperParamsResearch(
        CustomFCNN,
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
        ModelTrainingManagerComplex,
    )

    if debug:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats()
