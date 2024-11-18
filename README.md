# GMEGAN
Code for the GMEGAN method from the paper *"Monotone Generative Modeling via a Gromov-Monge Embedding"* by Lee, Yang, You, Lerman (2024).

**Paper link**: [https://arxiv.org/pdf/2311.01375](https://arxiv.org/pdf/2311.01375)

## Overview

This repository contains the implementation of the GMEGAN method for the CIFAR10 dataset. The main components include:

- `training_GMEGAN_CIFAR.py`: Python script for training GMEGAN on the CIFAR10 dataset.
- `transportmodules/`: Directory containing various neural network architectures for generators and discriminators used in GAN methods from the paper.

## How to Run

Follow these steps to run the code:

1. Clone the repository:
   ```bash
   git clone https://github.com/wonjunee/GMEGAN/
   ```

2. Run the Python script:
   ```bash
   python training_GMEGAN_CIFAR.py
   ```
