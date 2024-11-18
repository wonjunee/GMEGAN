# GMEGAN Code
This repository provides the code for the GMEGAN method introduced in the paper *"Monotone Generative Modeling via a Gromov-Monge Embedding"* by Lee, Yang, You, and Lerman (2024).

**Paper link**: [https://arxiv.org/pdf/2311.01375](https://arxiv.org/pdf/2311.01375)

## Overview

The repository contains the implementation of the GMEGAN method, with a focus on training it using the CIFAR10 dataset. Key components include:

- `training_GMEGAN_CIFAR.py`: Python script for training GMEGAN on the CIFAR10 dataset.
- `transportmodules/`: Directory containing neural network architectures for generators and discriminators used in the GAN methods described in the paper.

## How to Run

This code is designed to train GMEGAN on the CIFAR10 dataset. To use a different dataset, modify the `preprocessing_data` function and replace the dataset accordingly.

Steps to run the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/wonjunee/GMEGAN/
   ```

2. Execute the training script:
   ```bash
   python training_GMEGAN_CIFAR.py
   ```
