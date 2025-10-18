import logging
import time
from contextlib import nullcontext
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
import spare

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
    checkpoint = torch.load("/users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7", map_location='cpu')
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
        logger.info("test-time adaptation: SPARE (entropy-masked variant)")
        model = setup_spare(base_model)
    else:
        logger.info("Unknown adaptation; defaulting to source")
        model = setup_source(base_model)

    # evaluate on each severity and type of corruption in turn
    all_error = []
    accs_so_far = []  # for domain shift robustness
    prev_x = None
    prev_y = None
    prev_acc_at_time = None

    # Helper: format large numbers in base-10 scientific notation
    def fmt_sci(n: int) -> str:
        try:
            if n == 0:
                return "0"
            import math
            exp = int(math.floor(math.log10(abs(float(n)))))
            mant = float(n) / (10 ** exp)
            return f"{mant:.3f} x 10^{exp}"
        except Exception:
            return str(n)
    for severity in cfg.CORRUPTION.SEVERITY:
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            if i_c == 0:
                try:
                    if hasattr(model, 'reset'):
                        model.reset()
                        logger.info("")
                        logger.info("resetting model")
                except Exception:
                    logger.info("")
                    logger.warning("not resetting model")
            else:
                logger.info("")
                logger.warning("not resetting model")
            x_test, y_test = load_cifar10c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test = F.interpolate(x_test, size=(args.size, args.size),
                                   mode='bilinear', align_corners=False)
            # Ensure divisibility for SPARE patch size
            if cfg.MODEL.ADAPTATION == "REM":
                ps = cfg.SPARE.PATCH_SIZE
                if args.size % ps != 0:
                    raise ValueError(f"Input size {args.size} must be divisible by SPARE.PATCH_SIZE={ps}")
            # Optional dataset-level random pruning per corruption
            prune_mask = None
            if hasattr(cfg, 'SPARE') and cfg.SPARE.PRUNE_ENABLE and (cfg.SPARE.PRUNE_RANDOM_RANGE is not None):
                N = x_test.shape[0]
                a, b = cfg.SPARE.PRUNE_RANDOM_RANGE
                a = max(0, min(a, N))
                b = max(0, min(b, N))
                if b < a:
                    a, b = b, a
                k = int(torch.randint(low=a, high=b + 1, size=(1,)).item())
                # ensure at least 1 sample remains to evaluate
                if N > 0:
                    k = min(k, max(0, N - 1))
                idx = torch.randperm(N)[:k]
                prune_mask = torch.zeros(N, dtype=torch.bool)
                prune_mask[idx] = True  # True means pruned
                logger.info(f"dataset-prune: requested range=({cfg.SPARE.PRUNE_RANDOM_RANGE[0]}, {cfg.SPARE.PRUNE_RANDOM_RANGE[1]}), N={N}, k={k}, kept={N - k}")

            # Compute metrics (acc, nll, ece) with pruning awareness
            metrics = compute_metrics_prune_aware(
                model, x_test, y_test, cfg.TEST.BATCH_SIZE, device=device, prune_mask=prune_mask
            )
            acc, nll, ece, total_cnt, adapt_pruned_cnt, pred_pruned_cnt, std_below_low, std_above_high, mean_below_low, mean_above_high, adapt_time_total, adapt_macs_total = metrics
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
            logger.info(f"Error % [{corruption_type}{severity}]: {err:.2%}")
            logger.info(f"NLL [{corruption_type}{severity}]: {nll:.4f}")
            logger.info(f"ECE [{corruption_type}{severity}]: {ece:.4f}")
            # New metrics per corruption (averaged per corruption)
            # - Adaptation Time (s): total wall-clock time spent adapting; lower is better
            # - Adaptation MACs: total MACs for adapted samples; lower is better
            logger.info(f"Adaptation Time (lower is better) [{corruption_type}{severity}]: {adapt_time_total:.3f}s")
            logger.info(f"Adaptation MACs (lower is better) [{corruption_type}{severity}]: {fmt_sci(adapt_macs_total)}")

            # Domain Shift Robustness: std of accuracies across types so far (lower is better)
            accs_so_far.append(acc)
            if len(accs_so_far) >= 2:
                import math
                mean_acc = sum(accs_so_far) / float(len(accs_so_far))
                var_acc = sum((a - mean_acc) ** 2 for a in accs_so_far) / float(len(accs_so_far))
                dsr = math.sqrt(var_acc)
            else:
                dsr = 0.0
            logger.info(f"Domain Shift Robustness (std, lower is better) up to [{corruption_type}{severity}]: {dsr:.4f}")

            # Catastrophic Forgetting Rate (measured): re-evaluate previous corruption after current adaptation
            # CFR_current = max(0, prev_acc_at_time - prev_acc_after_current). Lower is better.
            if prev_x is not None and prev_y is not None and prev_acc_at_time is not None:
                try:
                    spare_model = model.module if hasattr(model, 'module') else model
                    ctx = spare_model.no_adapt_mode() if hasattr(spare_model, 'no_adapt_mode') else nullcontext()
                except Exception:
                    ctx = nullcontext()
                with ctx:
                    re_metrics = compute_metrics_prune_aware(
                        model, prev_x, prev_y, cfg.TEST.BATCH_SIZE, device=device, prune_mask=None
                    )
                    re_acc = re_metrics[0]
                cfr_measured = max(0.0, float(prev_acc_at_time) - float(re_acc))
                logger.info(f"Catastrophic Forgetting Rate (prev-domain, lower is better) after [{corruption_type}{severity}]: {cfr_measured:.4f}")
            # Update previous corruption cache for next iteration
            prev_x, prev_y, prev_acc_at_time = x_test, y_test, acc


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model


def compute_metrics_prune_aware(model: nn.Module,
                                x: torch.Tensor,
                                y: torch.Tensor,
                                batch_size: int = 100,
                                device: torch.device = None,
                                prune_mask: torch.Tensor = None):
    """Compute ACC, NLL, and ECE with optional dataset-level random pruning and model's internal pruning.

    prune_mask: optional bool tensor of shape [N], True means this sample is pruned from both adaptation and prediction.
    Returns (acc, nll, ece, total_cnt, adapt_pruned_total, pred_pruned_total, std_below_low, std_above_high, mean_below_low, mean_above_high)
    """
    if device is None:
        device = x.device
    if isinstance(device, str):
        device = torch.device(device)
    total_N = x.shape[0]
    n_batches = int((total_N + batch_size - 1) // batch_size)

    correct = 0
    total_eval = 0
    nll_sum = 0.0
    # For ECE, accumulate confidences and correctness
    confs_all = []
    correct_all = []
    adapt_pruned_total = 0
    pred_pruned_total = 0
    # SPARE prune category counts
    std_below_low = 0
    std_above_high = 0
    mean_below_low = 0
    mean_above_high = 0

    # Only compute/report prune categories when no dataset-level random pruning is used
    enable_prune_categories = prune_mask is None

    # Adaptation timing and MACs accumulators (averaged per corruption outside)
    adapt_time_total = 0.0
    adapt_macs_total = 0

    # Estimate per-image MACs for ViT-like models
    def estimate_vit_macs_per_image(stats_src, img_size: int) -> int:
        try:
            m = stats_src
            # Unwrap SPARE and DataParallel to reach underlying ViT
            if hasattr(m, 'module'):
                m = m.module
            if hasattr(m, 'model'):
                m = m.model
            if hasattr(m, 'module'):
                m = m.module
            # Extract key attributes
            # Patch size from conv proj
            ps = m.patch_embed.proj.weight.shape[2]
            D = getattr(m, 'embed_dim', m.head.in_features)
            depth = len(m.blocks)
            # heads from first block
            h = m.blocks[0].attn.num_heads
            # mlp ratio from first block
            D_m = m.blocks[0].mlp.fc1.out_features
            mlp_ratio = float(D_m) / float(D)
            # tokens length
            Ph = img_size // ps
            Pw = img_size // ps
            L = 1 + (Ph * Pw)
            C_in = m.patch_embed.proj.weight.shape[1]
            # MACs: Patch embedding conv
            macs_patch = (Ph * Pw) * (C_in * D * ps * ps)
            # Per-block MACs
            # QKV: 3 * L * D * D
            # Attn: 2 * L * L * D (QK^T and AV)
            # Out proj: L * D * D
            # MLP: 2 * L * D * (mlp_ratio * D)
            per_block = (3 * L * D * D) + (2 * L * L * D) + (L * D * D) + (2 * L * D * int(mlp_ratio * D))
            macs_blocks = depth * per_block
            # Head
            num_classes = m.head.out_features if hasattr(m.head, 'out_features') else 1000
            macs_head = D * num_classes
            total = macs_patch + macs_blocks + macs_head
            return int(total)
        except Exception:
            return 0

    with torch.no_grad():
        for b in range(n_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, total_N)
            x_b_full = x[start:end]
            y_b_full = y[start:end]

            # Move full batch to device for potential prediction
            x_b_full = x_b_full.to(device)
            y_b_full = y_b_full.to(device)

            # Prepare adaptation using dataset-level random pruning if provided
            pm_b = None
            keep_b = None
            per_img_macs = 0
            macs_counted = False
            if prune_mask is not None:
                pm_b = prune_mask[start:end]
                keep_b = (~pm_b)
                # Count samples removed from adaptation due to dataset-level random pruning
                adapt_pruned_total += int(pm_b.sum().item())

                # Run an adaptation pass on only the kept subset (if any)
                if keep_b.any():
                    x_adapt = x_b_full[keep_b]
                    # This call will perform adaptation (SPARE.forward uses enable_grad internally)
                    t0 = time.time()
                    _ = model(x_adapt)
                    adapt_time_total += (time.time() - t0)
                    # MACs only count adapted samples when prune_enable is set
                    stats_src = model.module if hasattr(model, 'module') else model
                    per_img_macs = estimate_vit_macs_per_image(stats_src, img_size=x_b_full.shape[-1])
                    adapt_macs_total += per_img_macs * int(x_adapt.shape[0])
                    macs_counted = True
                else:
                    # Nothing to adapt in this slice
                    pass

                # Accumulate SPARE prune categories based on the adaptation pass (only if enabled)
                stats_src = model.module if hasattr(model, 'module') else model
                if enable_prune_categories and hasattr(cfg, 'SPARE') and cfg.SPARE.PRUNE_ENABLE and \
                   hasattr(stats_src, '_last_delta_mean') and hasattr(stats_src, '_last_delta_std') and \
                   (stats_src._last_delta_mean is not None) and (stats_src._last_delta_std is not None):
                    try:
                        dm = torch.as_tensor(stats_src._last_delta_mean)
                        ds = torch.as_tensor(stats_src._last_delta_std)
                        std_below_low += int((ds < cfg.SPARE.PRUNE_TAU_LOW).sum().item())
                        std_above_high += int((ds > cfg.SPARE.PRUNE_TAU_HIGH).sum().item())
                        mean_below_low += int((dm < cfg.SPARE.PRUNE_MEAN_LOW).sum().item())
                        mean_above_high += int((dm > cfg.SPARE.PRUNE_MEAN_HIGH).sum().item())
                    except Exception:
                        pass

                # Prediction and metrics: either on full batch or kept subset depending on flag
                if cfg.SPARE.PRUNE_SKIP_PREDICTION:
                    if keep_b.any():
                        with (stats_src.no_adapt_mode() if hasattr(stats_src, 'no_adapt_mode') else torch.no_grad()):
                            output = model(x_b_full[keep_b])
                        # Count prediction prunes only when skipping prediction
                        pred_pruned_total += int(pm_b.sum().item())
                        y_eval = y_b_full[keep_b]
                    else:
                        # No kept samples to evaluate
                        continue
                else:
                    # Predict on full batch; do not count prediction prunes
                    with (stats_src.no_adapt_mode() if hasattr(stats_src, 'no_adapt_mode') else torch.no_grad()):
                        output = model(x_b_full)
                    y_eval = y_b_full
            else:
                # No dataset-level random pruning: single prediction pass (may adapt internally)
                t0 = time.time()
                output = model(x_b_full)
                adapt_time_total += (time.time() - t0)
                stats_src = model.module if hasattr(model, 'module') else model
                if enable_prune_categories and hasattr(cfg, 'SPARE') and cfg.SPARE.PRUNE_ENABLE and \
                   hasattr(stats_src, '_last_delta_mean') and hasattr(stats_src, '_last_delta_std') and \
                   (stats_src._last_delta_mean is not None) and (stats_src._last_delta_std is not None):
                    try:
                        dm = torch.as_tensor(stats_src._last_delta_mean)
                        ds = torch.as_tensor(stats_src._last_delta_std)
                        std_below_low += int((ds < cfg.SPARE.PRUNE_TAU_LOW).sum().item())
                        std_above_high += int((ds > cfg.SPARE.PRUNE_TAU_HIGH).sum().item())
                        mean_below_low += int((dm < cfg.SPARE.PRUNE_MEAN_LOW).sum().item())
                        mean_above_high += int((dm > cfg.SPARE.PRUNE_MEAN_HIGH).sum().item())
                    except Exception:
                        pass
                # Count MACs; if model returns keep_mask, count kept samples when pruning enabled
                per_img_macs = estimate_vit_macs_per_image(stats_src, img_size=x_b_full.shape[-1])
                # If pruning is disabled, adaptation used the whole batch; count all
                if not (hasattr(cfg, 'SPARE') and cfg.SPARE.PRUNE_ENABLE):
                    adapt_macs_total += per_img_macs * int(x_b_full.shape[0])
                    macs_counted = True
                else:
                    # If internal pruning is enabled, use _last_keep_mask to count kept samples
                    base = model.module if hasattr(model, 'module') else model
                    if hasattr(base, '_last_keep_mask') and base._last_keep_mask is not None:
                        kept = int(torch.as_tensor(base._last_keep_mask, device=x_b_full.device).sum().item())
                        adapt_macs_total += per_img_macs * kept
                        macs_counted = True
                y_eval = y_b_full

            # Handle outputs (either tuple (logits, keep_mask) or logits)
            if isinstance(output, tuple) and len(output) == 2:
                logits, keep_mask = output
                if keep_mask is not None and keep_mask.dtype == torch.bool:
                    kept = int(keep_mask.sum().item())
                    # Count adaptation pruning due to internal keep_mask
                    adapt_pruned = y_eval.shape[0] - kept
                    adapt_pruned_total += int(adapt_pruned)
                    # Count MACs for kept samples only when pruning enabled (avoid double counting)
                    if (not macs_counted) and per_img_macs:
                        adapt_macs_total += per_img_macs * kept
                        macs_counted = True
                    if logits.shape[0] != y_eval.shape[0]:
                        pred_pruned_total += int(y_eval.shape[0] - logits.shape[0])
                        y_sel = y_eval[keep_mask]
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
                        correct += (preds == y_eval).float().sum().item()
                        total_eval += y_eval.shape[0]
                        nll_sum += F.cross_entropy(logits, y_eval, reduction='sum').item()
                        probs = logits.softmax(dim=1)
                        confs = probs.max(dim=1).values
                        confs_all.append(confs.detach().cpu())
                        correct_all.append((preds == y_eval).detach().cpu())
                else:
                    logits = output[0]
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_eval).float().sum().item()
                    total_eval += y_eval.shape[0]
                    nll_sum += F.cross_entropy(logits, y_eval, reduction='sum').item()
                    probs = logits.softmax(dim=1)
                    confs = probs.max(dim=1).values
                    confs_all.append(confs.detach().cpu())
                    correct_all.append((preds == y_eval).detach().cpu())
            else:
                logits = output
                preds = logits.argmax(dim=1)
                correct += (preds == y_eval).float().sum().item()
                total_eval += y_eval.shape[0]
                nll_sum += F.cross_entropy(logits, y_eval, reduction='sum').item()
                probs = logits.softmax(dim=1)
                confs = probs.max(dim=1).values
                confs_all.append(confs.detach().cpu())
                correct_all.append((preds == y_eval).detach().cpu())

    if total_eval == 0:
        return 0.0, 0.0, 0.0, total_N, adapt_pruned_total, pred_pruned_total, None, None, None, None, adapt_time_total, adapt_macs_total

    acc = correct / total_eval
    nll = nll_sum / total_eval
    # compute ECE
    confs_all = torch.cat(confs_all) if len(confs_all) else torch.empty(0)
    correct_all = torch.cat(correct_all).float() if len(correct_all) else torch.empty(0)
    ece = compute_ece(confs_all, correct_all)
    # Only return prune categories when enabled; otherwise return None to suppress logging
    if enable_prune_categories:
        return acc, nll, ece, total_N, adapt_pruned_total, pred_pruned_total, std_below_low, std_above_high, mean_below_low, mean_above_high, adapt_time_total, adapt_macs_total
    else:
        return acc, nll, ece, total_N, adapt_pruned_total, pred_pruned_total, None, None, None, None, adapt_time_total, adapt_macs_total


def compute_ece(confs: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins on [0,1].
    confs: [N], correct: [N] in {0,1}
    """
    if confs.numel() == 0:
        return 0.0
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    for i in range(n_bins):
        lo = bin_boundaries[i]
        hi = bin_boundaries[i + 1]
        # include right edge in last bin
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


def setup_spare(model):
    model = spare.configure_model(model)
    params, param_names = spare.collect_params(model, ln_quarter=cfg.MODEL.LN_QUARTER)
    optimizer = setup_optimizer(params)
    rem_model = spare.SPARE(
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
        patch_size=cfg.SPARE.PATCH_SIZE,
        num_bins=cfg.SPARE.NUM_BINS,
        use_color_entropy=cfg.SPARE.USE_COLOR_ENTROPY,
        entropy_weight_power=cfg.SPARE.ENTROPY_WEIGHT_POWER,
        random_masking=cfg.SPARE.RANDOM_MASKING,
        num_squares=cfg.SPARE.NUM_SQUARES,
        mask_type=cfg.SPARE.MASK_TYPE,
        plot_loss=cfg.SPARE.PLOT_LOSS,
        plot_loss_path=cfg.SPARE.PLOT_LOSS_PATH,
        plot_ema_alpha=cfg.SPARE.PLOT_EMA_ALPHA,
        mcl_temperature=cfg.SPARE.MCL_TEMPERATURE,
        mcl_temperature_apply=cfg.SPARE.MCL_TEMPERATURE_APPLY,
        erl_activation=cfg.SPARE.ERL_ACTIVATION,
        erl_leaky_relu_slope=cfg.SPARE.ERL_LEAKY_RELU_SLOPE,
        erl_softplus_beta=cfg.SPARE.ERL_SOFTPLUS_BETA,
        disable_mcl=cfg.SPARE.DISABLE_MCL,
        disable_erl=cfg.SPARE.DISABLE_ERL,
        prune_random_range=cfg.SPARE.PRUNE_RANDOM_RANGE,
        # Adaptive masking from OPTIM
        m_adaptive_enable=cfg.OPTIM.M_ADAPTIVE_ENABLE,
        m_gap_target=cfg.OPTIM.M_GAP_TARGET,
        m_gap_kp=cfg.OPTIM.M_GAP_KP,
        m_min=cfg.OPTIM.M_MIN,
        m_max=cfg.OPTIM.M_MAX,
        m_adapt_smooth=cfg.OPTIM.M_ADAPT_SMOOTH,
        # Pruning options
        prune_enable=cfg.SPARE.PRUNE_ENABLE,
        prune_tau_low=cfg.SPARE.PRUNE_TAU_LOW,
        prune_tau_high=cfg.SPARE.PRUNE_TAU_HIGH,
        prune_mean_low=cfg.SPARE.PRUNE_MEAN_LOW,
        prune_mean_high=cfg.SPARE.PRUNE_MEAN_HIGH,
        prune_skip_prediction=cfg.SPARE.PRUNE_SKIP_PREDICTION,
    )
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return rem_model


if __name__ == '__main__':
    evaluate('CIFAR-10-C evaluation with SPARE (entropy-masked REM).')
