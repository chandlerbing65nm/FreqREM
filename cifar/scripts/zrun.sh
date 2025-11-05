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
cd /users/doloriel/work/Repo/SPARC/cifar

# python /users/doloriel/work/Repo/SPARC/cifar/scripts/rem_masking_trend_cifar10c.py \
#   --data_dir /scratch/project_465002264/datasets/cifar10c \
#   --ckpt_dir /users/doloriel/work/Repo/SPARC/ckpt \
#   --checkpoint /users/doloriel/work/Repo/SPARC/ckpt/vit_base_384_cifar10.t7 \
#   --out_dir /users/doloriel/work/Repo/SPARC/cifar/plots/REM \
#   --num_examples 100 \
#   --severity 5 \
#   --batch_size 20 \
#   --progression 0 100 10 \
#   --save_mask_examples 1 \
#   --mask_example_levels 0 10 20 \
#   --mask_figs_dir /users/doloriel/work/Repo/SPARC/cifar/figs/REM

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------

python /users/doloriel/work/Repo/SPARC/cifar/scripts/spare_masking_trend_cifar10c.py \
  --data_dir /scratch/project_465002264/datasets/cifar10c \
  --ckpt_dir /users/doloriel/work/Repo/SPARC/ckpt \
  --checkpoint /users/doloriel/work/Repo/SPARC/ckpt/vit_base_384_cifar10.t7 \
  --batch_size 50 \
  --num_examples 100 \
  --severity 5 \
  --out_dir /users/doloriel/work/Repo/SPARC/cifar/plots/SPARE/Trend/Spectral \
  --progression 0 100 10 \
  --save_mask_examples 2 \
  --mask_example_levels 0 15 30 \
  --mask_figs_dir /users/doloriel/work/Repo/SPARC/cifar/figs/SPARE/Spectral \
  --example_class airplane \
  --random_seed 50 \
  --save_frequency_energy_plot \
  --mask_type spectral

python /users/doloriel/work/Repo/SPARC/cifar/scripts/spare_masking_trend_cifar10c.py \
  --data_dir /scratch/project_465002264/datasets/cifar10c \
  --ckpt_dir /users/doloriel/work/Repo/SPARC/ckpt \
  --checkpoint /users/doloriel/work/Repo/SPARC/ckpt/vit_base_384_cifar10.t7 \
  --out_dir /users/doloriel/work/Repo/SPARC/cifar/plots/SPARE/Trend/Spatial \
  --num_examples 100 \
  --severity 5 \
  --batch_size 20 \
  --progression 0 100 10 \
  --save_mask_examples 2 \
  --mask_example_levels 0 15 30 \
  --mask_figs_dir /users/doloriel/work/Repo/SPARC/cifar/figs/SPARE/Spatial \
  --patch_size 8 \
  --masking_mode random \
  --random_seed 50 \
  --example_class airplane \
  --save_spatial_entropy_plot \
  --mask_type spatial