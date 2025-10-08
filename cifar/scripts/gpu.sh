#!/bin/bash

#SBATCH --job-name=chandlertasks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465002264
#SBATCH --output=logs/output_%j.txt

# Activate conda in non-interactive shells and activate the env
source /scratch/project_465002264/miniconda3/etc/profile.d/conda.sh
conda activate rem

# Set the working directory
cd /users/doloriel/work/Repo/FreqREM/cifar

# python /users/doloriel/work/Repo/FreqREM/cifar/scripts/masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/FreqREM/ckpt \
#   --checkpoint /users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7 \
#   --out_dir /users/doloriel/work/Repo/FreqREM/cifar/plots/FreqREM \
#   --num_examples 1000 \
#   --severity 5 \
#   --batch_size 50 \
#   --mode phase \
#   --progression 25 50 5 \
#   --save_phase_examples 1 \
#   --phase_example_levels 0 25 30 \
#   --figs_dir /users/doloriel/work/Repo/FreqREM/cifar/figs

# Example: Combined phase-mix then patch masking (uses same t for both)
python /users/doloriel/work/Repo/FreqREM/cifar/scripts/phase_mix_then_mask_trend_cifar10c.py \
  --data_dir /scratch/project_465002264/datasets/cifar10c \
  --ckpt_dir /users/doloriel/work/Repo/FreqREM/ckpt \
  --checkpoint /users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7 \
  --out_dir /users/doloriel/work/Repo/FreqREM/cifar/plots/PhaseMixThenMask \
  --num_examples 20 \
  --severity 5 \
  --batch_size 20 \
  --progression 0 100 10 \
  --phase_alpha 0.6 \
  --save_mix_examples 1 \
  --mix_example_levels 0 25 50 75 100 \
  --figs_dir /users/doloriel/work/Repo/FreqREM/cifar/figs/PhaseMixThenMask \
  --corruptions gaussian_noise defocus_blur snow jpeg_compression
