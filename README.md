# SPARE: Sample Pruning And Random Erasing for Continual Test-Time Adaptation

This repository contains code for SPARE (Selective Pruning And Random Erasing), built on top of the SPARE variant in `cifar/spare.py` and evaluation scripts for CIFAR-C.

The CIFAR runners are:
- `cifar/cifar10c_vit_spare.py`
- `cifar/cifar100c_vit_spare.py`

These wrap a base ViT checkpoint with the SPARE adaptation (`REM` in configs) and add selective pruning and random erasing controls via the `SPARE` config options in `cifar/conf.py` and the YAML files under `cifar/cfgs/`.

## Installation

See INSTALL.md for detailed environment setup:
- [INSTALL.md](INSTALL.md)

Quick summary (ROCm example is in INSTALL.md): create a conda env, install PyTorch/torchvision, then install project requirements from `requirements.txt`.

## Data: CIFAR-C

Download the CIFAR-C datasets and note the directory you place them in (pass as `--data_dir` when running):
- CIFAR-10-C: https://zenodo.org/records/2535967
- CIFAR-100-C: https://zenodo.org/records/3555552

RobustBench loaders in `cifar/cifar10c_vit_spare.py` and `cifar/cifar100c_vit_spare.py` will read from `--data_dir`.

## CIFAR Experiments

Below is a minimal setup and the exact commands to reproduce SPARE on CIFAR-10-C and CIFAR-100-C.

### Environment setup

```bash
conda init
conda activate rem
cd home/cifar
```

If you are inside the repository root, the `cifar/` folder is at `Repo/SPARE/cifar/`.

### CIFAR-10 → CIFAR-10-C

Run the following from inside the `cifar/` directory (so that paths like `cfgs/...` resolve):

```bash
python -m cifar10c_vit_spare \
     --cfg cfgs/cifar10/spare.yaml \
     --data_dir data_path \
     --patch_size 8 \
     --lr 0.001 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking \
     --num_squares 1 \
     --mask_type binary \
     --m 0.10 --n 3 \
     --prune_enable
```

### CIFAR-100 → CIFAR-100-C

```bash
python -m cifar100c_vit_spare \
     --cfg cfgs/cifar100/spare.yaml \
     --data_dir data_path \
     --patch_size 8 \
     --lr 0.0001 \
     --lamb 1.0 \
     --margin 0.0 \
     --random_masking \
     --num_squares 1 \
     --mask_type binary \
     --m 0.10 --n 3 \
     --prune_enable
```

### Notes

- Checkpoints:
  - CIFAR-10: `cifar/cifar10c_vit_spare.py` loads a ViT checkpoint from `/users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7`.
  - CIFAR-100: `cifar/cifar100c_vit_spare.py` loads a checkpoint from `/users/doloriel/work/Repo/SPARE/ckpt/pretrain_cifar100.t7`.
  - If your checkpoints are elsewhere, update those paths in the scripts or place the files accordingly.
- Input size and patch size:
  - The default input resize is `--size 384` (see `cifar/conf.py`). If using SPARE masking, the input size must be divisible by `--patch_size` (e.g., 384 divisible by 8).
- Config knobs:
  - YAMLs under `cifar/cfgs/cifar10/spare.yaml` and `cifar/cfgs/cifar100/spare.yaml` set defaults for learning rate, masking schedule (`m`, `n`), and SPARE options. CLI flags override the YAML.

## ImageNet Experiments

Coming soon. Contents to be added later.

## Acknowledgements

This codebase builds upon and was inspired by the following works and repositories:

1. REM (Ranked Entropy Minimization)
   - Repository: https://github.com/pilsHan/rem.git
   - Citation:
     
     ```text
     @article{Han2025RankedEM,
       title={Ranked Entropy Minimization for Continual Test-Time Adaptation},
       author={Jisu Han and Jaemin Na and Wonjun Hwang},
       journal={ArXiv},
       year={2025},
       volume={abs/2505.16441},
       url={https://api.semanticscholar.org/CorpusID:278788718}
     }
     ```

2. Continual-MAE
   - Repository: https://github.com/RanXu2000/continual-mae.git
   - Citation:
     
     ```text
     @article{Liu2023ContinualMAEAD,
       title={Continual-MAE: Adaptive Distribution Masked Autoencoders for Continual Test-Time Adaptation},
       author={Jiaming Liu and Ran Xu and Senqiao Yang and Renrui Zhang and Qizhe Zhang and Zehui Chen and Yandong Guo and Shanghang Zhang},
       journal={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
       year={2023},
       pages={28653-28663},
       url={https://api.semanticscholar.org/CorpusID:266374852}
     }
     ```