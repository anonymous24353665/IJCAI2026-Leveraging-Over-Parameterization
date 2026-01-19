import argparse
import cProfile
import pstats
from pathlib import Path
import sys
import os

# add the repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Generators.ArchitecturesGenerator.src.train_conv import run_training

DEBUG = False  # Abilita il profiling se True

def main():
    parser = argparse.ArgumentParser(description="Launcher per il training di reti convoluzionali shallow")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["MNIST", "FMNIST"],
        help="Dataset da usare (MNIST o FMNIST)"
    )

    # -------- COSTRUZIONE PATH ASSOLUTO --------
    # BASE_DIR = root del progetto (modifica l'indice dei parents se necessario)
    BASE_DIR = Path(__file__).resolve().parents[0]

    # Path assoluto al file di configurazione
    config_path_ini = (
            BASE_DIR
            / "src"
            / "configs"
            / f"config.ini"
    )

    config_path_yaml = (
            BASE_DIR
            / "src"
            / "configs"
            / "training_config.yaml"
    )

    parser.add_argument(
        "--hidden_layers_dim",
        type=int,
        nargs="+",
        default=[5, 15, 25, 50, 100, 200, 500],
        help="Dimensione dei layer fully connected dopo la convoluzione"
    )

    parser.add_argument(
        "--config_ini",
        type=Path,
        default=config_path_ini,
        help="Percorso del file di configurazione ini"
    )
    parser.add_argument(
        "--config_yaml",
        type=Path,
        default=config_path_yaml,
        help="Percorso del file di configurazione yaml"
    )


    parser.add_argument("--min_increment", type=float, default=0.1, help="Valore minimo per la ricerca binaria")
    parser.add_argument("--max_increment", type=float, default=9, help="Valore massimo per la ricerca binaria")
    parser.add_argument("--steps_limit", type=int, default=10, help="Numero massimo di step per la ricerca binaria")

    parser.add_argument("--rs_loss_bool", action="store_true", help="Se usare il Regularization Sensitivity Loss")
    parser.add_argument("--skip_binary_search", action="store_true", help="Se saltare la ricerca binaria e trainare direttamente")

    args = parser.parse_args()

    if DEBUG:
        profiler = cProfile.Profile()
        profiler.enable()

    run_training(
        dataset_name=args.dataset,
        hidden_layers_dim=args.hidden_layers_dim,
        config_path_ini=args.config_ini,
        config_path_yaml=args.config_yaml,
        max_increment=args.max_increment,
        min_increment=args.min_increment,
        steps_limit=args.steps_limit,
        rs_loss_bool=args.rs_loss_bool,
        skip_binary_search=args.skip_binary_search
    )

    if DEBUG:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumulative")
        stats.print_stats()


if __name__ == "__main__":
    main()
