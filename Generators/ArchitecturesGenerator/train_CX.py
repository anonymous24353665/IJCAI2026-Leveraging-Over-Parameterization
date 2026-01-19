import argparse
from pathlib import Path

from Generators.ArchitecturesGenerator.one_rs_param.train_2_layers import run_training



def main():
    parser = argparse.ArgumentParser(description="Run NN training experiments")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MNIST", "FMNIST"],
        help="Dataset name",
    )

    # -------- COSTRUZIONE PATH ASSOLUTO --------
    # BASE_DIR = root del progetto (modifica l'indice dei parents se necessario)
    BASE_DIR = Path(__file__).resolve().parents[0]

    # Path assoluto al file di configurazione
    config_path_ini = (
        BASE_DIR
        / "one_rs_param"
        / "configs"
        / f"config.ini"
    )

    config_path_yaml = (
            BASE_DIR
            / "one_rs_param"
            / "configs"
            / "training_config.yaml"
    )

    parser.add_argument(
        "--config_ini",
        type=str,
        default=str(config_path_ini),  # path assoluto di default
        help="Path to configuration file",
    )

    parser.add_argument(
        "--config_yaml",
        type=str,
        default=str(config_path_yaml),  # path assoluto di default
        help="Path to configuration file",
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500, 1000, 2000],
        help="Hidden layer dimensions",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable profiling",
    )

    args = parser.parse_args()

    run_training(
        dataset_name=args.dataset,
        hidden_layers_dim=args.hidden_dims,
        config_path_ini=args.config_ini,
        config_path_yaml=args.config_yaml,
        debug=args.debug,
        rs_loss_bool=False,
        skip_binary_search=True
    )


if __name__ == "__main__":
    main()
