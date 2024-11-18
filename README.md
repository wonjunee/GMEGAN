# GMEGAN Code

This repository contains the code for the GMEGAN method introduced in the paper *"Monotone Generative Modeling via a Gromov-Monge Embedding"* by Lee, Yang, You, and Lerman (2024).

**Paper link**: [https://arxiv.org/pdf/2311.01375](https://arxiv.org/pdf/2311.01375)

## Overview

This repository provides implementations of the GMEGAN method and other GAN-based methods. It supports training on both artificial datasets (e.g., mixtures of Gaussians) and the CIFAR10 dataset. Key components include:

- `training_multiple_GAN_gaussian.py`: Python script for training multiple GAN-based methods on an artificial dataset.
- `training_GMEGAN_CIFAR.py`: Python script for training GMEGAN on the CIFAR10 dataset.
- `transportmodules/`: Directory containing neural network architectures for generators and discriminators used in the GAN methods described in the paper.

## How to Run

### Training on an Artificial Dataset (Mixture of Gaussians)

This script runs multiple GAN-based methods simultaneously on an artificial dataset comprising a mixture of Gaussians. The included GAN-based methods are:

- **GMEGAN (Ours)**  
- **GAN**  
  Ian Goodfellow et al., *Generative Adversarial Nets*, 2014  
- **WGAN**  
  Martin Arjovsky, Soumith Chintala, and Léon Bottou, *Wasserstein Generative Adversarial Networks*, 2017  
- **WGAN-GP**  
  Ishaan Gulrajani et al., *Improved Training of Wasserstein GANs*, 2017  
- **WDIV**  
  Jiqing Wu et al., *Wasserstein Divergence for GANs*, 2018  
- **OTM**  
  Litu Rout, Alexander Korotin, and Evgeny Burnaev, *Generative Modeling with Optimal Transport Maps*, 2022  
- **VAEGAN**  
  Anders Boesen Lindbo Larsen et al., *Autoencoding Beyond Pixels Using a Learned Similarity Metric*, 2016  
- **VEEGAN**  
  Akash Srivastava et al., *VEEGAN: Reducing Mode Collapse in GANs Using Implicit Variational Learning*, 2017

#### Steps to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/wonjunee/GMEGAN/
   ```

2. Run the training script:
   ```bash
   python training_multiple_GAN_gaussian.py
   ```

### Training on an Image Dataset (CIFAR10)

The GMEGAN code is preconfigured for training on the CIFAR10 dataset. To train on a different dataset, modify the `preprocessing_data` function to replace the CIFAR10 dataset with your desired dataset.

#### Steps to Run the Code

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

GMEGAN is copyrighted by Regents of the University of Minnesota and Duke Kunshan University and covered by US 63/715,441. Regents of the University of Minnesota and Duke Kunshan University will license the use of GMEGAN solely for educational and research purposes by non-profit institutions and government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.
