import logging
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from robustbench.data import load_cifar100c
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
    # CIFAR-100 checkpoint
    checkpoint = torch.load("/users/doloriel/work/Repo/FreqREM/ckpt/pretrain_cifar100.t7", map_location='cpu')
    checkpoint = rm_substr_from_state_dict(checkpoint['model'], 'module.')
    base_model.load_state_dict(checkpoint, strict=True)
    del checkpoint

    if cfg.TEST.ckpt is not None:
        if device.type == 'cuda':
            base_model = torch.nn.DataParallel(base_model)
        ckpt = torch.load(cfg.TEST.ckpt, map_location='cpu')
        state2 = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        base_model.load_state_dict(state2, strict=False)
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
            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                            severity, cfg.DATA_DIR, False,
                                            [corruption_type])
            x_test = F.interpolate(x_test, size=(args.size, args.size),
                                   mode='bilinear', align_corners=False)
            # Ensure divisibility for EntREM patch size
            if cfg.MODEL.ADAPTATION == "REM":
                ps = cfg.ENTREM.PATCH_SIZE
                if args.size % ps != 0:
                    raise ValueError(f"Input size {args.size} must be divisible by ENTREM.PATCH_SIZE={ps}")
            # Optional dataset-level random pruning per corruption
            prune_mask = None
            if hasattr(cfg, 'ENTREM') and cfg.ENTREM.PRUNE_ENABLE and (cfg.ENTREM.PRUNE_RANDOM_RANGE is not None):
                N = x_test.shape[0]
                a, b = cfg.ENTREM.PRUNE_RANDOM_RANGE
                a = max(0, min(a, N))
                b = max(0, min(b, N))
                if b < a:
                    a, b = b, a
                k = int(torch.randint(low=a, high=b + 1, size=(1,)).item())
                idx = torch.randperm(N)[:k]
                prune_mask = torch.zeros(N, dtype=torch.bool)
                prune_mask[idx] = True

            metrics = compute_metrics_prune_aware(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device, prune_mask=prune_mask
            )
            acc, nll, ece, total_cnt, adapt_pruned_cnt, pred_pruned_cnt, std_below_low, std_above_high, mean_below_low, mean_above_high = metrics
            logger.info(
                f"counts [{corruption_type}{severity}] total={total_cnt}, not_in_adapt={adapt_pruned_cnt}, not_in_pred={pred_pruned_cnt}"
            )
            if std_below_low is not None:
                logger.info(
                    f"prune categories [{corruption_type}{severity}] "
                    f"sigma<tau_l={std_below_low}, sigma>tau_u={std_above_high}, "
                    f"mu<mu_l={mean_below_low}, mu>mu_u={mean_above_high}"
                )
            err = 1. - acc
            all_error.append(err)
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")


def setup_source(model):
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def compute_metrics_prune_aware(model: nn.Module,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                batch_size: int = 100,
                                device: torch.device = None,
                                prune_mask: torch.Tensor = None):
    """Compute ACC, NLL, and ECE with optional dataset-level random pruning and model's internal pruning."""
    if device is None:
        device = x.device
    if isinstance(device, str):
        device = torch.device(device)
    total_N = x.shape[0]
    n_batches = int((total_N + batch_size - 1) // batch_size)

    correct = 0
    total_eval = 0
    nll_sum = 0.0
    confs_all = []
    correct_all = []
    adapt_pruned_total = 0
    pred_pruned_total = 0
    std_below_low = 0
    std_above_high = 0
    mean_below_low = 0
    mean_above_high = 0

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, total_N)
            x_b = x[start:end]
            y_b = y[start:end]
            if prune_mask is not None:
                pm_b = prune_mask[start:end]
                keep_b = (~pm_b)
                adapt_pruned_total += int(pm_b.sum().item())
                pred_pruned_total += int(pm_b.sum().item())
                if not keep_b.any():
                    continue
                x_b = x_b[keep_b]
                y_b = y_b[keep_b]
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            output = model(x_b)
            stats_src = model.module if hasattr(model, 'module') else model
            if hasattr(cfg, 'ENTREM') and cfg.ENTREM.PRUNE_ENABLE and \
               hasattr(stats_src, '_last_delta_mean') and hasattr(stats_src, '_last_delta_std') and \
               (stats_src._last_delta_mean is not None) and (stats_src._last_delta_std is not None):
                try:
                    dm = torch.as_tensor(stats_src._last_delta_mean)
                    ds = torch.as_tensor(stats_src._last_delta_std)
                    std_below_low += int((ds < cfg.ENTREM.PRUNE_TAU_LOW).sum().item())
                    std_above_high += int((ds > cfg.ENTREM.PRUNE_TAU_HIGH).sum().item())
                    mean_below_low += int((dm < cfg.ENTREM.PRUNE_MEAN_LOW).sum().item())
                    mean_above_high += int((dm > cfg.ENTREM.PRUNE_MEAN_HIGH).sum().item())
                except Exception:
                    pass

            if isinstance(output, tuple) and len(output) == 2:
                logits, keep_mask = output
                if keep_mask is not None and keep_mask.dtype == torch.bool:
                    kept = int(keep_mask.sum().item())
                    adapt_pruned = y_b.shape[0] - kept
                    adapt_pruned_total += int(adapt_pruned)
                    if logits.shape[0] != y_b.shape[0]:
                        pred_pruned_total += int(y_b.shape[0] - logits.shape[0])
                        y_sel = y_b[keep_mask]
                        if y_sel.numel() == 0:
                            continue
                        preds = logits.argmax(dim=1)
                        correct += (preds == y_sel).float().sum().item()
                        total_eval += logits.shape[0]
                        nll_sum += F.cross_entropy(logits, y_sel, reduction='sum').item()
                        probs = logits.softmax(dim=1)
                        confs = probs.max(dim=1).values
                        confs_all.append(confs.detach().cpu())
                        correct_all.append((preds == y_sel).detach().cpu())
                    else:
                        preds = logits.argmax(dim=1)
                        correct += (preds == y_b).float().sum().item()
                        total_eval += y_b.shape[0]
                        nll_sum += F.cross_entropy(logits, y_b, reduction='sum').item()
                        probs = logits.softmax(dim=1)
                        confs = probs.max(dim=1).values
                        confs_all.append(confs.detach().cpu())
                        correct_all.append((preds == y_b).detach().cpu())
                else:
                    logits = output[0]
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_b).float().sum().item()
                    total_eval += y_b.shape[0]
                    nll_sum += F.cross_entropy(logits, y_b, reduction='sum').item()
                    probs = logits.softmax(dim=1)
                    confs = probs.max(dim=1).values
                    confs_all.append(confs.detach().cpu())
                    correct_all.append((preds == y_b).detach().cpu())
            else:
                logits = output
                preds = logits.argmax(dim=1)
                correct += (preds == y_b).float().sum().item()
                total_eval += y_b.shape[0]
                nll_sum += F.cross_entropy(logits, y_b, reduction='sum').item()
                probs = logits.softmax(dim=1)
                confs = probs.max(dim=1).values
                confs_all.append(confs.detach().cpu())
                correct_all.append((preds == y_b).detach().cpu())

    if total_eval == 0:
        return 0.0, 0.0, 0.0, total_N, adapt_pruned_total, pred_pruned_total, None, None, None, None

    acc = correct / total_eval
    nll = nll_sum / total_eval
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    return acc, nll, ece, total_N, adapt_pruned_total, pred_pruned_total, std_below_low, std_above_high, mean_below_low, mean_above_high


def compute_ece(confs: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    if confs.numel() == 0:
        return 0.0
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confs >= lo) & (confs <= hi)
        else:
            in_bin = (confs >= lo) & (confs < hi)
        count = in_bin.sum().item()
        if count == 0:
            continue
        conf_bin = confs[in_bin].mean().item()
        acc_bin = correct[in_bin].mean().item()
        prop = count / confs.numel()
        ece += abs(acc_bin - conf_bin) * prop
    return float(ece)


def setup_optimizer(params):
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
    params, _ = entrem.collect_params(model, ln_quarter=cfg.MODEL.LN_QUARTER)
    optimizer = setup_optimizer(params)
    rem_model = entrem.EntREM(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        m=cfg.OPTIM.M,
        n=cfg.OPTIM.N,
        lamb=cfg.OPTIM.LAMB,
        margin=cfg.OPTIM.MARGIN,
        m_step=cfg.OPTIM.M_STEP,
        m_top=cfg.OPTIM.M_TOP,
        m_progress_enable=cfg.OPTIM.M_PROGRESS_ENABLE,
        m_progress_dir=cfg.OPTIM.M_PROGRESS_DIR,
        patch_size=cfg.ENTREM.PATCH_SIZE,
        num_bins=cfg.ENTREM.NUM_BINS,
        use_color_entropy=cfg.ENTREM.USE_COLOR_ENTROPY,
        entropy_weight_power=cfg.ENTREM.ENTROPY_WEIGHT_POWER,
        random_masking=cfg.ENTREM.RANDOM_MASKING,
        num_squares=cfg.ENTREM.NUM_SQUARES,
        mask_type=cfg.ENTREM.MASK_TYPE,
        plot_loss=cfg.ENTREM.PLOT_LOSS,
        plot_loss_path=cfg.ENTREM.PLOT_LOSS_PATH,
        plot_ema_alpha=cfg.ENTREM.PLOT_EMA_ALPHA,
        mcl_temperature=cfg.ENTREM.MCL_TEMPERATURE,
        mcl_temperature_apply=cfg.ENTREM.MCL_TEMPERATURE_APPLY,
        erl_activation=cfg.ENTREM.ERL_ACTIVATION,
        erl_leaky_relu_slope=cfg.ENTREM.ERL_LEAKY_RELU_SLOPE,
        erl_softplus_beta=cfg.ENTREM.ERL_SOFTPLUS_BETA,
        disable_mcl=cfg.ENTREM.DISABLE_MCL,
        disable_erl=cfg.ENTREM.DISABLE_ERL,
        prune_random_range=cfg.ENTREM.PRUNE_RANDOM_RANGE,
        # Pruning options
        prune_enable=cfg.ENTREM.PRUNE_ENABLE,
        prune_tau_low=cfg.ENTREM.PRUNE_TAU_LOW,
        prune_tau_high=cfg.ENTREM.PRUNE_TAU_HIGH,
        prune_mean_low=cfg.ENTREM.PRUNE_MEAN_LOW,
        prune_mean_high=cfg.ENTREM.PRUNE_MEAN_HIGH,
        prune_skip_prediction=cfg.ENTREM.PRUNE_SKIP_PREDICTION,
    )
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rem_model


if __name__ == '__main__':
    evaluate('CIFAR-100-C evaluation with EntREM (entropy-masked REM).')
