import os
import sys
import warnings

import torch

# =========================
# Filtraggio log TensorFlow/CUDA
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # nasconde info e warning
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disabilita ottimizzazioni oneDNN
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'



# =========================
# Device PyTorch
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# =========================
# Configurazioni base
# =========================
DATASET_NAMES = ["MNIST", "FMNIST", "CIFAR10", "CUSTOM_CIFAR10"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'results'))
BACKUP_FOLDER = os.path.join(RESULTS_FOLDER, "BACKUP")

# Creazione cartelle generali
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)

# =========================
# Funzione per creare file solo se non esiste
# =========================
def create_results_file(file_path):
    # Assicurati che la cartella padre esista
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "x") as f:
            pass  # file vuoto creato solo se non esiste
    except FileExistsError:
        pass

# =========================
# Creazione cartelle specifiche dataset e CSV
# =========================
for DATASET_NAME in DATASET_NAMES:

    # Cartelle principali per regularized e not_regularized
    REG_FOLDER = os.path.join(RESULTS_FOLDER, DATASET_NAME, "regularized_models")
    NOT_REG_FOLDER = os.path.join(RESULTS_FOLDER, DATASET_NAME, "not_regularized_models")

    # Creazione cartelle principali
    os.makedirs(REG_FOLDER, exist_ok=True)
    os.makedirs(NOT_REG_FOLDER, exist_ok=True)

    # Sottocartelle: best_models / all_models / refined_attempt
    for subfolder in ["best_models", "all_models", "refined_attempt"]:
        os.makedirs(os.path.join(REG_FOLDER, subfolder), exist_ok=True)
        os.makedirs(os.path.join(NOT_REG_FOLDER, subfolder), exist_ok=True)

    # CSV files
    CSV_FILE_BEST_CANDIDATES_REGULARIZED = os.path.join(REG_FOLDER, "results_best_candidates.csv")
    CSV_FILE_BEST_CANDIDATES_NOT_REGULARIZED = os.path.join(NOT_REG_FOLDER, "results_best_candidates.csv")
    CSV_FILE_ALL_CANDIDATES_REGULARIZED = os.path.join(REG_FOLDER, "results_all_candidates.csv")
    CSV_FILE_ALL_CANDIDATES_NOT_REGULARIZED = os.path.join(NOT_REG_FOLDER, "results_all_candidates.csv")

    # Creazione file CSV solo se non esistono
    create_results_file(CSV_FILE_BEST_CANDIDATES_REGULARIZED)
    create_results_file(CSV_FILE_BEST_CANDIDATES_NOT_REGULARIZED)
    create_results_file(CSV_FILE_ALL_CANDIDATES_REGULARIZED)
    create_results_file(CSV_FILE_ALL_CANDIDATES_NOT_REGULARIZED)
