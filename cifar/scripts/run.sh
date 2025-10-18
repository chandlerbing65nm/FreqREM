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
cd /users/doloriel/work/Repo/SPARE/cifar

# python /users/doloriel/work/Repo/SPARE/cifar/scripts/masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/SPARE/ckpt \
#   --checkpoint /users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7 \
#   --out_dir /users/doloriel/work/Repo/SPARE/cifar/plots/REM \
#   --num_examples 100 \
#   --severity 5 \
#   --batch_size 20 \
#   --progression 0 100 10 \
#   --save_mask_examples 1 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/SPARE/cifar/figs/REM

# python /users/doloriel/work/Repo/SPARE/cifar/scripts/entropy_freq_masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/SPARE/ckpt \
#   --checkpoint /users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7 \
#   --batch_size 50 \
#   --num_examples 100 \
#   --severity 5 \
#   --out_dir /users/doloriel/work/Repo/SPARE/cifar/plots/SPARE \
#   --progression 0 100 10 \
#   --save_mask_examples 2 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/SPARE/cifar/figs/SPARE \
#   --example_class airplane

# python /users/doloriel/work/Repo/SPARE/cifar/scripts/entropy_masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/SPARE/ckpt \
#   --checkpoint /users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7 \
#   --out_dir /users/doloriel/work/Repo/SPARE/cifar/plots/SPARE \
#   --num_examples 100 \
#   --severity 5 \
#   --batch_size 20 \
#   --progression 0 100 10 \
#   --save_mask_examples 3 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/SPARE/cifar/figs/SPARE \
#   --patch_size 8 \
#   --masking_mode random \
#   --random_seed 42 \
#   --example_class airplane


python cifar/scripts/entropy_random_token_masking_trend_cifar10c.py \
  --data_dir /scratch/project_465002264/datasets/cifar10c \
  --ckpt_dir /users/doloriel/work/Repo/SPARE/ckpt \
  --checkpoint /users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7 \
  --out_dir /users/doloriel/work/Repo/SPARE/cifar/plots/TokenREM \
  --num_examples 100 \
  --severity 5 \
  --batch_size 20 \
  --progression 0 100 10 \
  --save_mask_examples 3 \
  --mask_example_levels 0 10 20 \
  --mask_figs_dir /users/doloriel/work/Repo/SPARE/cifar/figs/TokenREM \
  --patch_size 8 \
  --masking_mode random \
  --random_seed 42 \
  --example_class airplane