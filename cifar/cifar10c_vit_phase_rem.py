import logging

import torch
import torch.optim as optim

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import torch.nn as nn
from conf import cfg, load_cfg_fom_args
from collections import OrderedDict

from rem_phase import REMPhase
import rem  # reuse configure_model / collect_params helpers

logger = logging.getLogger(__name__)


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def evaluate(description):
    args = load_cfg_fom_args(description)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions)
    checkpoint = torch.load("/users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7", map_location='cpu')
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
    base_model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    if cfg.TEST.ckpt is not None:
        # make parallel only if CUDA is available
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
        checkpoint = torch.load(cfg.TEST.ckpt, map_location='cpu')
        base_model.load_state_dict(checkpoint['model'], strict=False)
    else:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
    base_model.to(device)

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    else:
        logger.info("test-time adaptation: REMPhase (phase-distortion variants)")
        model = setup_rem_phase(base_model)

    # evaluate on each severity and type of corruption in turn
    All_error = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    model.reset()
                    logger.info("resetting model")
                except Exception:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = torch.nn.functional.interpolate(x_test, size=(args.size, args.size), \
                mode='bilinear', align_corners=False)
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device.type)
            err = 1. - acc
            All_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer(params):
    """Set up optimizer for test-time adaptation (same as REM)."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


def setup_rem_phase(model):
    # Reuse same parameterization as REM (LayerNorms etc.)
    model = rem.configure_model(model)
    params, param_names = rem.collect_params(model)
    optimizer = setup_optimizer(params)
    # Levels and seed from config (fallback to defaults inside REMPhase if missing)
    levels = tuple(cfg.PHASE.LEVELS) if hasattr(cfg, 'PHASE') and hasattr(cfg.PHASE, 'LEVELS') else (0.0, 0.25, 0.30)
    phase_seed = cfg.PHASE.SEED if hasattr(cfg, 'PHASE') and hasattr(cfg.PHASE, 'SEED') else None
    rem_model = REMPhase(model, optimizer,
                         steps=cfg.OPTIM.STEPS,
                         episodic=cfg.MODEL.EPISODIC,
                         levels=levels,
                         lamb=cfg.OPTIM.LAMB,
                         margin=cfg.OPTIM.MARGIN,
                         phase_seed=phase_seed)
    logger.info(f"optimizer for adaptation: {optimizer}")
    return rem_model


if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation with REMPhase.')
