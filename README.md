# BLOSAM: Bilevel Optimization-based Sharpness-Aware Minimization


BLOSAM (**Bilevel Optimization-based Sharpness-Aware Minimization**) is an advanced optimization framework designed to improve the generalization performance of deep neural networks. 

It reformulates **SAM (Sharpness-Aware Minimization)** as a **bilevel optimization problem**, modeling perturbation dynamics as fast variables and parameter dynamics as slow variables. 

This separation enables smoother convergence trajectories, reduced sensitivity to perturbation radius ρ, and improved adaptability to non-stationary training environments.

Reformulates SAM into a **bilevel optimization framework**.
- Enhances **generalization** across vision and NLP tasks.
- Reduces sensitivity to **radius parameter ρ.
- Compatible with standard optimizers (SGD, SAM, etc.).
- Simple PyTorch implementation for easy integration.

### Basic Training Command
Use the following format to run training:
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python3 trainasam.py --arch <model> --dataset <dataset>
