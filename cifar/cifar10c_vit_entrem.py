import logging
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

from conf import cfg, load_cfg_fom_args
import entrem

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
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.') if isinstance(checkpoint, dict) else checkpoint
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        base_model.load_state_dict(checkpoint['model'], strict=True)
    else:
        base_model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    # Apply potential adaptation checkpoint (optional)
    if cfg.TEST.ckpt is not None:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
        ckpt = torch.load(cfg.TEST.ckpt, map_location='cpu')
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        base_model.load_state_dict(state, strict=False)
    else:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
    base_model.to(device)

    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE (source)")
        model = setup_source(base_model)
    elif cfg.MODEL.ADAPTATION == "REM":
        logger.info("test-time adaptation: EntREM (entropy-masked variant)")
        model = setup_entrem(base_model)
    else:
        logger.info("Unknown adaptation; defaulting to source")
        model = setup_source(base_model)

    # evaluate on each severity and type of corruption in turn
    all_error = []
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    if hasattr(model, 'reset'):
                        model.reset()
                        logger.info("resetting model")
                except Exception:
                    logger.warning("not resetting model")
            else:
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = F.interpolate(x_test, size=(args.size, args.size),
                                   mode='bilinear', align_corners=False)
            # Ensure divisibility for EntREM patch size
            if cfg.MODEL.ADAPTATION == "REM":
                ps = cfg.ENTREM.PATCH_SIZE
                if args.size % ps != 0:
                    raise ValueError(f"Input size {args.size} must be divisible by ENTREM.PATCH_SIZE={ps}")
            acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device.type)
            err = 1. - acc
            all_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def setup_optimizer(params):
    """Set up optimizer for test-time adaptation.

    For best results, try tuning the learning rate and batch size.
    """
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


def setup_entrem(model):
    model = entrem.configure_model(model)
    params, param_names = entrem.collect_params(model)
    optimizer = setup_optimizer(params)
    rem_model = entrem.EntREM(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        m=cfg.OPTIM.M,
        n=cfg.OPTIM.N,
        lamb=cfg.OPTIM.LAMB,
        margin=cfg.OPTIM.MARGIN,
        patch_size=cfg.ENTREM.PATCH_SIZE,
        num_bins=cfg.ENTREM.NUM_BINS,
        use_color_entropy=cfg.ENTREM.USE_COLOR_ENTROPY,
        entropy_weight_power=cfg.ENTREM.ENTROPY_WEIGHT_POWER,
    )
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rem_model


if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation with EntREM (entropy-masked REM).')
