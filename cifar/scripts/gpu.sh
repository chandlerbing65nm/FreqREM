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
#   --out_dir /users/doloriel/work/Repo/FreqREM/cifar/plots/REM \
#   --num_examples 100 \
#   --severity 5 \
#   --batch_size 20 \
#   --progression 0 100 10 \
#   --save_mask_examples 1 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/FreqREM/cifar/figs/REM

# python /users/doloriel/work/Repo/FreqREM/cifar/scripts/entropy_masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/FreqREM/ckpt \
#   --checkpoint /users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7 \
#   --out_dir /users/doloriel/work/Repo/FreqREM/cifar/plots/FreqREM \
#   --num_examples 100 \
#   --severity 5 \
#   --batch_size 20 \
#   --progression 0 100 10 \
#   --save_mask_examples 3 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/FreqREM/cifar/figs/FreqREM \
#   --patch_size 8 \
#   --entropy_bins 32 \
#   --use_color_entropy \
#   --entropy_weight_power 2 \
#   --masking_mode random \
#   --random_seed 42 \
#   --example_class airplane

# Ranked entropy three-view evaluation (view0, view1_low, view2_high)
python /users/doloriel/work/Repo/FreqREM/cifar/scripts/ranked_entropy_views_cifar10c.py \
  --data_dir /scratch/project_465002264/datasets/cifar10c \
  --ckpt_dir /users/doloriel/work/Repo/FreqREM/ckpt \
  --checkpoint /users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7 \
  --out_dir /users/doloriel/work/Repo/FreqREM/cifar/plots \
  --num_examples 100 \
  --severity 5 \
  --batch_size 20 \
  --patches_per_side 16 \
  --entropy_bins 32 \
  --use_color_entropy \
  --mask_percent 10 \
  --save_examples 3 \
  --figs_dir /users/doloriel/work/Repo/FreqREM/cifar/figs \
  --example_class airplane
