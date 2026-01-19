# Leveraging Over-Parameterization to Improve the Verifiability of Neural Networks

This repository contains the code associated with the paper:  
**"Leveraging Over-Parameterization to Improve the Verifiability of Neural Networks"** (IJCAI 2026).

---

## Overview

This repository provides:

- Training scripts for different neural network architectures (SH, CV, CX, CF).  
- Support for MNIST, Fashion-MNIST, and CIFAR datasets pre-filtered from a ResNet18.  
- Ability to train networks under over-parameterization regime with or without regularization (`rs_loss`).  
- Scripts to generate VNNLIB properties for local robustness verification.  

All experiments can be run inside a **Docker container** to ensure reproducibility.

---

## Docker Setup

### Build the container

From the root of the repository:

```bash
docker build -t ijcai_image .
```

### Run the container

Mount your local repo or datasets folder if needed:

```bash
docker run -it -v /path/to/local/repo:/app ijcai_image
```

Run the entrypoint:

```bash
./docker-entrypoint.sh
```

Inside the container, navigate to:

```bash
cd /app/Generators/ArchitecturesGenerator
```

---

## Training Neural Networks

You can train **SH, CV, CF, or CX architectures**, either with regularization (`rs_loss`) or without.

### Examples:

#### MNIST / Fashion-MNIST

Train with rs_loss regularization:

```bash
python train_SH.py --dataset MNIST|FMNIST --rs_loss_bool
python train_CX.py --dataset MNIST|FMNIST --rs_loss_bool
python train_CV.py --dataset MNIST|FMNIST --rs_loss_bool
```

Train baseline architectures without regularization:

```bash
python train_SH.py --dataset MNIST|FMNIST --skip_binary_search
python train_CX.py --dataset MNIST|FMNIST --skip_binary_search
python train_CV.py --dataset MNIST|FMNIST --skip_binary_search

```

#### CIFAR (Custom Dataset)

Train with rs_loss:

```bash
python train_CF.py --rs_loss_bool
```

Train baseline architecture without regularization:

```bash
python train_CF.py --skip_binary_search
```

> Replace `SH`, `CV`, `CX`, `CF` with the architecture you want to train.

---

Results are saved in `/app/Generators/ArchitecturesGenerator/DATASET_NAME/regularized_models` or `not_regularized_models`

## Generating VNNLIB Properties

To generate local robustness properties:

```bash
cd /app/Generators/PropertyGenerator
```

```bash
python generate_vnnlib_properties.py
python generate_vnnlib_properties_cifar.py
```

Properties are saved in `/app/Generators/PropertyGenerator/results`

- This script generates `.vnnlib` files for the CIFAR_CUSTOM10 dataset.  
- By default, it generates 100 properties.

---

## Data on Google Drive
Given the significant computational time required to train all architectures, we provide the trained models used in the paper at the following link:
https://drive.google.com/drive/folders/1WwTIEEOz9FEcLgCcn9ST9UZV7Owb-9PU

## Notes

- All scripts are configured to run inside the Docker container.  
- Python environment and dependencies are installed via Conda inside the container.  

---

