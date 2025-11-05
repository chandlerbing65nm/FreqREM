# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS) for SPARC: Stochastic Patch Erasing with
Adaptive Residual Correction for Continual Test-Time Adaptation (CTTA), and related baselines."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Standard'

# Choice of (source, norm, tent)
# - source: baseline without adaptation
# - norm: test-time normalization
# - tent: test-time entropy minimization (ours)
_C.MODEL.ADAPTATION = 'source'

# By default tent is online, with updates persisting across batches.
# To make adaptation episodic, and reset the model for each batch, choose True.
_C.MODEL.EPISODIC = False
 
# LayerNorm adaptation scope across transformer blocks (for ViT backbones)
# Choices: 'default' (original behavior), 'q1','q2','q3','q4','all'
_C.MODEL.LN_QUARTER = 'default'

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5, 4, 3, 2, 1]

# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.CORRUPTION.NUM_EX = 10000

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# Masking ratio
_C.OPTIM.KEEP = 288

# COTTA
_C.OPTIM.MT = 0.999
_C.OPTIM.RST = 0.01
_C.OPTIM.AP = 0.92

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./output"

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Continual_MAE
_C.block_size = 16
_C.mask_ratio = 0.5
_C.use_hog = False
_C.hog_ratio = 1

# ViDA
_C.TEST.vida_rank1 = 1
_C.TEST.vida_rank2 = 128
_C.OPTIM.MT_ViDA = 0.999
_C.OPTIM.ViDALR=1e-4

# REM parameters
_C.OPTIM.M = 0.1
_C.OPTIM.N = 3
_C.OPTIM.LAMB = 1.0
_C.OPTIM.MARGIN = 0.0

# Phase distortion options
_C.PHASE = CfgNode()
_C.PHASE.LEVELS = [0.0, 0.25, 0.30]
_C.PHASE.SEED = None
_C.PHASE.ALPHA = 0.45
_C.PHASE.CHANNEL_ORDER = [0, 1, 2]
_C.PHASE.CHANNEL_STEPS = [0, 1, 2, 3]
_C.PHASE.USE_MCL = True
_C.PHASE.USE_ERL = True
_C.PHASE.CONSISTENCY_MODE = 'mcl'
_C.PHASE.CWAL_THRESHOLD = 0.7

# (Removed phase-mix-then-mask config)

# SPARC: Stochastic Patch Erasing with Adaptive Residual Correction for
# Continual Test-Time Adaptation (CTTA) options
_C.SPARC = CfgNode()
_C.SPARC.PATCH_SIZE = 16
_C.SPARC.RANDOM_MASKING = False
_C.SPARC.NUM_SQUARES = 1
_C.SPARC.MASK_TYPE = 'binary'  # choices: 'binary', 'gaussian', 'mean'
_C.SPARC.PLOT_LOSS = False
_C.SPARC.PLOT_LOSS_PATH = ""
_C.SPARC.PLOT_EMA_ALPHA = 0.98
_C.SPARC.MCL_TEMPERATURE = 1.0
_C.SPARC.MCL_TEMPERATURE_APPLY = 'both'  # choices: 'teacher', 'student', 'both'
_C.SPARC.MCL_DISTANCE = 'ce'             # choices: 'ce','kl','js','mse','mae'
_C.SPARC.ERL_ACTIVATION = 'relu'         # choices: 'relu','leaky_relu','softplus','gelu','sigmoid','identity'
_C.SPARC.ERL_LEAKY_RELU_SLOPE = 0.01
_C.SPARC.ERL_SOFTPLUS_BETA = 1.0
_C.SPARC.DISABLE_MCL = False
_C.SPARC.DISABLE_ERL = False
# MARN (Manifold-Aware Ranked Normalization)
# (TALN options removed)
_C.SPARC.LOGSPARC_ENABLE = 'none'           # choices: 'none','gamma','beta','gammabeta'
_C.SPARC.LOGSPARC_LR_MULT = 1.0             # LR multiplier for Logsparc parameters
_C.SPARC.LOGSPARC_REG = 0.0                 # regularizer strength for monotonic gamma/beta
_C.SPARC.LOGSPARC_TEMP = 0.0                # if >0, apply as temperature to beta pre-softplus; masked views only
_C.SPARC.LOGSPARC_TYPE2 = False             # if true and LOGSPARC_ENABLE!=none: gamma/beta are [B,C]
_C.SPARC.LOGSPARC_TYPE3 = False             # if true and LOGSPARC_ENABLE!=none: gamma/beta are [1,C]
_C.SPARC.LOGSPARC_ABL_TYPE1 = False         # Ablation: remove L2 norm after gamma/beta transform
_C.SPARC.LOGSPARC_ABL_TYPE2 = False         # Ablation: only apply L2 norm, skip gamma/beta transform
_C.SPARC.LOGSPARC_ABL_TYPE3 = False         # Ablation: also apply to unmasked (m=0) logits

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    parser.add_argument("--index", default=1, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--size", default=384, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--unc_thr", default=0.05, type=float)
    #parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--data_dir", type=str, default='/mnt/work1/cotta/continual-mae/cifar/data/')

    # LayerNorm adaptation scope
    parser.add_argument("--ln_quarter", type=str, default=None,
                        choices=['default','q1','q2','q3','q4','all'],
                        help="Which transformer block quarter's LayerNorms to adapt (ViT): default|q1|q2|q3|q4|all")

    parser.add_argument("--use_hog", action="store_true",
                    help="if use hog")
    parser.add_argument("--hog_ratio", type=float,
                    help="hog ratio")

    # SPARC (CTTA) optimization CLI options
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of adaptation updates per batch (maps to OPTIM.STEPS)")
    parser.add_argument("--m", type=float, default=None,
                        help="Masking increment per level in [0,1] (maps to OPTIM.M)")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of masking levels (maps to OPTIM.N)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for optimizer (maps to OPTIM.LR)")
    # (Removed progressive/adaptive masking CLI args)
    parser.add_argument("--lamb", type=float, default=None,
                        help="Lambda for entropy-ordering loss (maps to OPTIM.LAMB)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Margin multiplier in entropy-ordering loss (maps to OPTIM.MARGIN)")

    # SPARC-specific CLI options
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Patch size for entropy-based masking (default from cfg)")
    # (Removed: --num_bins, --entropy_bins, --entropy_levels, --use_color_entropy, --entropy_weight_power)
    parser.add_argument("--random_masking", action="store_true",
                        help="Use random grid-aligned square placement instead of entropy centroid")
    parser.add_argument("--num_squares", type=int, default=None,
                        help="Number of equal-size squares to place per masking level (default from cfg)")
    parser.add_argument("--mask_type", type=str, default=None, choices=['binary', 'gaussian', 'mean'],
                        help="How to fill masked regions: 'binary' (zero), 'gaussian' (blurred), or 'mean' (per-image mean)")
    # (Removed SPARC pruning CLI options)
    # SPARC plotting CLI options
    parser.add_argument("--plot_loss", action="store_true",
                        help="If set, save a PNG plot of EMA of MCL and ERL across steps")
    parser.add_argument("--plot_loss_path", type=str, default=None,
                        help="Output path for the loss plot PNG")
    parser.add_argument("--plot_ema_alpha", type=float, default=None,
                        help="EMA alpha for smoothing MCL/ERL curves (e.g., 0.98)")
    # Disable specific losses
    parser.add_argument("--disable_mcl", action="store_true",
                        help="Disable the Mask Consistency Loss (MCL) term")
    parser.add_argument("--disable_erl", action="store_true",
                        help="Disable the Entropy Ranking Loss (ERL) term")
    # MCL temperature
    parser.add_argument("--mcl_temperature", type=float, default=None,
                        help="Temperature for MCL softmax/log_softmax (default from cfg)")
    parser.add_argument("--mcl_temperature_apply", type=str, default=None, choices=['teacher','student','both'],
                        help="Where to apply MCL temperature: teacher, student, or both")
    parser.add_argument("--mcl_distance", type=str, default=None,
                        choices=['ce','kl','js','mse','mae'],
                        help="Distance metric for MCL: ce|kl|js|mse|mae (default: ce)")
    # ERL activation options
    parser.add_argument("--erl_activation", type=str, default=None,
                        choices=['relu','leaky_relu','softplus','gelu','sigmoid','identity'],
                        help="Activation function used in ERL margin term")
    parser.add_argument("--erl_leaky_relu_slope", type=float, default=None,
                        help="Negative slope for LeakyReLU when erl_activation=leaky_relu")
    parser.add_argument("--erl_softplus_beta", type=float, default=None,
                        help="Beta parameter for Softplus when erl_activation=softplus")
    # Logsparc CLI options
    parser.add_argument("--logsparc_enable", type=str, default=None,
                        choices=['none','gamma','beta','gammabeta'],
                        help="Enable Logsparc on logits with mode: none|gamma|beta|gammabeta")
    parser.add_argument("--logsparc_lr_mult", type=float, default=None,
                        help="Learning rate multiplier for Logsparc parameters (default from cfg)")
    parser.add_argument("--logsparc_reg", type=float, default=None,
                        help="Monotonicity regularizer strength for gamma/beta across masking levels")
    parser.add_argument("--logsparc_temp", type=float, default=None,
                        help="If > 0, apply as temperature to beta pre-softplus; only when mask ratio > 0")
    parser.add_argument("--logsparc_type2", action="store_true",
                        help="If set with --logsparc_enable, use per-class gamma/beta of shape [B,C]")
    parser.add_argument("--logsparc_type3", action="store_true",
                        help="If set with --logsparc_enable, use global per-class gamma/beta of shape [1,C]")
    # Logsparc ablation CLI options (work in combination with --logsparc_enable)
    parser.add_argument("--logspark_abl_type1", action="store_true",
                        help="Ablation: remove L2 normalization after gamma/beta transformation")
    parser.add_argument("--logspark_abl_type2", action="store_true",
                        help="Ablation: only apply L2 normalization to logits; do NOT apply gamma/beta transform")
    parser.add_argument("--logspark_abl_type3", action="store_true",
                        help="Ablation: apply the chosen transform/normalization policy to unmasked (m=0) logits as well")
    # (TALN CLI options removed)
    # (Removed Phase-mix-then-mask CLI arg)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.size = args.size
    cfg.DATA_DIR = args.data_dir
    cfg.TEST.ckpt = args.checkpoint
    # Map CLI seed to config if provided; otherwise keep YAML/default
    if args.seed is not None:
        cfg.RNG_SEED = int(args.seed)

    # Model LayerNorm selection from CLI
    if args.ln_quarter is not None:
        cfg.MODEL.LN_QUARTER = args.ln_quarter.lower()

    # Populate OPTIM from CLI if provided
    if args.steps is not None:
        cfg.OPTIM.STEPS = args.steps
    if args.m is not None:
        cfg.OPTIM.M = args.m
    if args.n is not None:
        cfg.OPTIM.N = args.n
    if args.lr is not None:
        cfg.OPTIM.LR = args.lr
    # (Removed progressive/adaptive OPTIM overrides from CLI)
    if args.lamb is not None:
        cfg.OPTIM.LAMB = args.lamb
    if args.margin is not None:
        cfg.OPTIM.MARGIN = args.margin

    cfg.use_hog = args.use_hog
    cfg.hog_ratio = args.hog_ratio

    # Populate SPARC config from CLI if provided
    if args.patch_size is not None:
        cfg.SPARC.PATCH_SIZE = args.patch_size
    # (Removed: num_bins/entropy_bins/entropy_levels/use_color_entropy/entropy_weight_power)
    cfg.SPARC.RANDOM_MASKING = bool(args.random_masking)
    if args.num_squares is not None:
        cfg.SPARC.NUM_SQUARES = max(1, int(args.num_squares))
    if args.mask_type is not None:
        cfg.SPARC.MASK_TYPE = str(args.mask_type).lower()
    # Plotting options
    if args.plot_loss:
        cfg.SPARC.PLOT_LOSS = True
    if args.plot_loss_path is not None:
        cfg.SPARC.PLOT_LOSS_PATH = args.plot_loss_path
    if args.plot_ema_alpha is not None:
        cfg.SPARC.PLOT_EMA_ALPHA = args.plot_ema_alpha
    # Disable flags
    if args.disable_mcl:
        cfg.SPARC.DISABLE_MCL = True
    if args.disable_erl:
        cfg.SPARC.DISABLE_ERL = True
    # MCL temperature
    if args.mcl_temperature is not None:
        cfg.SPARC.MCL_TEMPERATURE = args.mcl_temperature
    if args.mcl_temperature_apply is not None:
        cfg.SPARC.MCL_TEMPERATURE_APPLY = args.mcl_temperature_apply.lower()
    if args.mcl_distance is not None:
        cfg.SPARC.MCL_DISTANCE = args.mcl_distance.lower()
    # ERL activation
    if args.erl_activation is not None:
        cfg.SPARC.ERL_ACTIVATION = args.erl_activation.lower()
    if args.erl_leaky_relu_slope is not None:
        cfg.SPARC.ERL_LEAKY_RELU_SLOPE = args.erl_leaky_relu_slope
    if args.erl_softplus_beta is not None:
        cfg.SPARC.ERL_SOFTPLUS_BETA = args.erl_softplus_beta
    # (TALN options removed)
    # Logsparc options
    if args.logsparc_enable is not None:
        cfg.SPARC.LOGSPARC_ENABLE = args.logsparc_enable.lower()
    if args.logsparc_lr_mult is not None:
        cfg.SPARC.LOGSPARC_LR_MULT = float(args.logsparc_lr_mult)
    if args.logsparc_reg is not None:
        cfg.SPARC.LOGSPARC_REG = float(args.logsparc_reg)
    if args.logsparc_temp is not None:
        cfg.SPARC.LOGSPARC_TEMP = float(args.logsparc_temp)
    if args.logsparc_type2:
        cfg.SPARC.LOGSPARC_TYPE2 = True
    if args.logsparc_type3:
        cfg.SPARC.LOGSPARC_TYPE3 = True
    # Logsparc ablations
    if args.logspark_abl_type1:
        cfg.SPARC.LOGSPARC_ABL_TYPE1 = True
    if args.logspark_abl_type2:
        cfg.SPARC.LOGSPARC_ABL_TYPE2 = True
    if args.logspark_abl_type3:
        cfg.SPARC.LOGSPARC_ABL_TYPE3 = True
    # (Feature polynomial toggles removed)
    # (Removed LN sub-option gating as stats/affine/EMA are removed)
    # (Removed SPARC pruning CLI mappings)

    # (Removed PHASEMIX override from CLI)


    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.RNG_SEED)
    except Exception:
        pass
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info("Using RNG seed: %d", cfg.RNG_SEED)
    logger.info(cfg)
    return args
