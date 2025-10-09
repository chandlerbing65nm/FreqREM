# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

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

# Phase-mix-then-mask options
_C.PHASEMIX = CfgNode()
_C.PHASEMIX.ALPHA = 1.0

# Entropy-based REM (EntREM) options
_C.ENTREM = CfgNode()
_C.ENTREM.PATCH_SIZE = 16
_C.ENTREM.NUM_BINS = 32
_C.ENTREM.LEVELS = [0, 10, 20]
_C.ENTREM.USE_COLOR_ENTROPY = False
_C.ENTREM.ENTROPY_WEIGHT_POWER = 2.0

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
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--size", default=384, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--unc_thr", default=0.05, type=float)
    #parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--data_dir", type=str, default='/mnt/work1/cotta/continual-mae/cifar/data/')

    parser.add_argument("--use_hog", action="store_true",
                    help="if use hog")
    parser.add_argument("--hog_ratio", type=float,
                    help="hog ratio")

    # REM/EntREM optimization CLI options
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of adaptation updates per batch (maps to OPTIM.STEPS)")
    parser.add_argument("--m", type=float, default=None,
                        help="Masking increment per level in [0,1] (maps to OPTIM.M)")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of masking levels (maps to OPTIM.N)")
    parser.add_argument("--lamb", type=float, default=None,
                        help="Lambda for entropy-ordering loss (maps to OPTIM.LAMB)")
    parser.add_argument("--margin", type=float, default=None,
                        help="Margin multiplier in entropy-ordering loss (maps to OPTIM.MARGIN)")

    # EntREM-specific CLI options
    parser.add_argument("--patch_size", type=int, default=None,
                        help="Patch size for entropy-based masking (default from cfg)")
    parser.add_argument("--num_bins", type=int, default=None,
                        help="Histogram bins for entropy computation (default from cfg)")
    parser.add_argument("--entropy_bins", type=int, default=None,
                        help="Alias for --num_bins: histogram bins for entropy computation")
    parser.add_argument("--entropy_levels", type=int, nargs='+', default=None,
                        help="Masking levels in percent for entropy-based masking, e.g., 0 10 20")
    parser.add_argument("--use_color_entropy", action="store_true",
                        help="Compute entropy over RGB channels (averaged) instead of grayscale")
    parser.add_argument("--entropy_weight_power", type=float, default=None,
                        help="Power for weighting top entropies when computing centroid (>1 emphasizes higher entropies)")
    # Phase-mix-then-mask CLI options
    parser.add_argument("--phase_mix_alpha", type=float, default=None,
                        help="Alpha in [0,1] for phase mix (1.0=magnitude-only counterpart)")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    cfg.size = args.size
    cfg.DATA_DIR = args.data_dir
    cfg.TEST.ckpt = args.checkpoint

    # Populate OPTIM from CLI if provided
    if args.steps is not None:
        cfg.OPTIM.STEPS = args.steps
    if args.m is not None:
        cfg.OPTIM.M = args.m
    if args.n is not None:
        cfg.OPTIM.N = args.n
    if args.lamb is not None:
        cfg.OPTIM.LAMB = args.lamb
    if args.margin is not None:
        cfg.OPTIM.MARGIN = args.margin

    cfg.use_hog = args.use_hog
    cfg.hog_ratio = args.hog_ratio

    # Populate EntREM config from CLI if provided
    if args.patch_size is not None:
        cfg.ENTREM.PATCH_SIZE = args.patch_size
    if args.num_bins is not None:
        cfg.ENTREM.NUM_BINS = args.num_bins
    if args.entropy_bins is not None:
        cfg.ENTREM.NUM_BINS = args.entropy_bins
    if args.entropy_levels is not None:
        cfg.ENTREM.LEVELS = args.entropy_levels
    # Booleans / floats
    cfg.ENTREM.USE_COLOR_ENTROPY = bool(args.use_color_entropy)
    if args.entropy_weight_power is not None:
        cfg.ENTREM.ENTROPY_WEIGHT_POWER = args.entropy_weight_power

    # Populate PHASEMIX from CLI if provided
    if args.phase_mix_alpha is not None:
        cfg.PHASEMIX.ALPHA = args.phase_mix_alpha


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
    # torch.cuda.manual_seed(seed)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
    return args
