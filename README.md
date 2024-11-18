# GMEGAN Code

This repository contains the code for the GMEGAN method introduced in the paper *"Monotone Generative Modeling via a Gromov-Monge Embedding"* by Lee, Yang, You, and Lerman (2024).

**Paper link**: [https://arxiv.org/pdf/2311.01375](https://arxiv.org/pdf/2311.01375)

## Overview

This repository implements the GMEGAN method, designed for training on the CIFAR10 dataset. The main components are:

- `training_GMEGAN_CIFAR.py`: Python script for training GMEGAN on the CIFAR10 dataset.
- `transportmodules/`: Directory containing neural network architectures for generators and discriminators used in the GAN methods described in the paper.

## How to Run

The code is configured to train GMEGAN on the CIFAR10 dataset. To train on a different dataset, update the `preprocessing_data` function to replace the CIFAR10 dataset with your dataset.

### Steps to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/wonjunee/GMEGAN/
   ```

2. Run the training script:
   ```bash
   python training_GMEGAN_CIFAR.py
   ```

## Copyright and License Notice
© 2024 Regents of the University of Minnesota and Duke Kunshan University

GMEGAN is copyrighted by Regents of the University of Minnesota and Duke Kunshan University and covered by US 63/715,441. Regents of the University of Minnesota and Duke Kunshan University will license the use of [GMEGAN] solely for educational and research purposes by non-profit institutions and government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
