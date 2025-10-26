# Environment Setup

This guide summarizes how to create a working ROCm-based environment for the SPARC repository.

## 1. Create Conda Environment

```bash
conda create -n rem python=3.8
conda activate rem
```

## 2. Install Project Requirements


```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/rocm5.6
pip install -r requirements.txt
```

The provided `requirements.txt` already includes the necessary versions and fetches RobustBench & AutoAttack from GitHub.

## 4. Validate Installation

```bash
python -c "import torch, numpy, timm, robustbench; print(torch.__version__, numpy.__version__)"
```

If imports succeed without warnings, the environment is ready for REM scripts under `cifar/` and `imagenet/`.
