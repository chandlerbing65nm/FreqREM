from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List
from contextlib import contextmanager


class Entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between current logits and a detached target distribution."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Cross-entropy with temperature scaling: softmax(logits/temperature).

    Args:
        x: current logits [B, K]
        x_ema: target logits [B, K] (detached)
        temperature: > 0 scalar
    """
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -((x_ema / t).softmax(1) * (x / t).log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp_teacher(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling applied to teacher (target) only."""
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -((x_ema / t).softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_temp_student(x: torch.Tensor, x_ema: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature scaling applied to student (current) only."""
    t = torch.tensor(temperature, dtype=x.dtype, device=x.device)
    t = torch.clamp(t, min=1e-6)
    return -(x_ema.softmax(1) * (x / t).log_softmax(1)).sum(1)


def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                             model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model: nn.Module) -> nn.Module:
    """Enable grads where needed. Keep BatchNorm in special mode as in REM."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model


def collect_params(model: nn.Module, ln_quarter: str = 'default'):
    """Collect trainable parameters (LayerNorm weights/bias) with optional quarter selection.

    Quarter selection applies to ViT-style transformer blocks named like 'blocks.X'.
    Options:
      - 'default': original policy (skip LayerNorms in blocks 9,10,11 and any top-level 'norm')
      - 'q1'|'q2'|'q3'|'q4': only LayerNorms inside the corresponding quarter of transformer blocks
      - 'all': LayerNorms inside all transformer blocks

    For CNN backbones, the original exclusions (skip 'layer4', skip top-level 'norm') still apply.
    """
    ln_quarter = str(ln_quarter).lower()
    params = []
    names = []

    # First pass: gather transformer block indices if present
    block_indices = set()
    for nm, _ in model.named_modules():
        if 'blocks.' in nm:
            try:
                after = nm.split('blocks.')[1]
                idx_str = after.split('.')[0]
                if idx_str.isdigit():
                    block_indices.add(int(idx_str))
            except Exception:
                pass

    sorted_blocks = sorted(block_indices)
    n_blocks = len(sorted_blocks)

    def quarter_index_ranges(n: int):
        # Returns list of (start_inclusive, end_inclusive) for 4 quarters dividing [0, n)
        # Use rounding to distribute remainder evenly.
        bounds = [int(round(i * n / 4.0)) for i in range(5)]  # 0, ~n/4, ~n/2, ~3n/4, n
        return [(bounds[i], bounds[i + 1] - 1) for i in range(4)]

    allowed_blocks = None  # None means use default policy
    if ln_quarter in ['q1', 'q2', 'q3', 'q4', 'all'] and n_blocks > 0:
        if ln_quarter == 'all':
            allowed_blocks = set(sorted_blocks)
        else:
            q_map = {'q1': 0, 'q2': 1, 'q3': 2, 'q4': 3}
            q_idx = q_map[ln_quarter]
            ranges = quarter_index_ranges(n_blocks)
            # Map back to actual block indices using their sorted order
            start_pos, end_pos = ranges[q_idx]
            start_pos = max(0, min(start_pos, n_blocks - 1))
            end_pos = max(start_pos, min(end_pos, n_blocks - 1))
            allowed_blocks = set(sorted_blocks[start_pos:end_pos + 1])

    # Second pass: collect LayerNorm parameters according to policy
    for nm, m in model.named_modules():
        # Exclusions common to both policies
        if 'layer4' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        # Determine if this module is inside a specific transformer block
        this_block_idx = None
        if 'blocks.' in nm:
            try:
                after = nm.split('blocks.')[1]
                idx_str = after.split('.')[0]
                if idx_str.isdigit():
                    this_block_idx = int(idx_str)
            except Exception:
                pass

        # Apply selection policy
        if allowed_blocks is None:
            # default policy: skip LNs in blocks 9,10,11 (ViT-B typical last quarter)
            if any(f'blocks.{k}' in nm for k in ['9', '10', '11']):
                continue
        else:
            # quarter/all policy: only allow LNs in allowed transformer blocks
            if this_block_idx is None or this_block_idx not in allowed_blocks:
                continue

        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    # x: [B,3,H,W] in [0,1]; return [B,1,H,W]
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b)


def compute_patch_entropy_map(x_img: torch.Tensor,
                              patches_per_side: int,
                              num_bins: int) -> torch.Tensor:
    """
    Compute Shannon entropy per patch over a grid.
    x_img: [B,C,H,W] in [0,1], C can be 1 (grayscale) or 3 (RGB)
    Returns: [B, patches_per_side, patches_per_side]
    """
    B, C, H, W = x_img.shape
    patch_h = H // patches_per_side
    patch_w = W // patches_per_side
    assert H % patches_per_side == 0 and W % patches_per_side == 0, "H and W must be divisible by patches_per_side"

    patches = x_img.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)  # [B,C,Ph,Pw,patch_h,patch_w]
    Ph = patches.size(2)
    Pw = patches.size(3)
    patch_size = patch_h * patch_w
    patches = patches.contiguous().view(B, C, Ph, Pw, patch_size)  # [B,C,Ph,Pw,S]
    patches_flat = patches.view(B * C * Ph * Pw, patch_size)  # [N, S]

    q = torch.clamp((patches_flat * float(num_bins)).long(), 0, num_bins - 1)  # [N, S]
    ones = torch.ones_like(q, dtype=torch.float32)
    counts = torch.zeros(q.size(0), num_bins, device=x_img.device, dtype=torch.float32)
    counts.scatter_add_(1, q, ones)

    probs = counts / float(patch_size)  # [N, num_bins]
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)  # [N]

    ent = ent.view(B, C, Ph, Pw)
    ent = ent.mean(dim=1)  # [B,Ph,Pw]
    return ent


def build_centered_square_mask(H: int, W: int, side: int, cy: int, cx: int) -> torch.Tensor:
    y0 = max(0, min(H - side, cy - side // 2))
    x0 = max(0, min(W - side, cx - side // 2))
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[y0:y0 + side, x0:x0 + side] = 1.0
    return mask


def _gaussian_kernel1d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    xs = torch.arange(kernel_size, device=device, dtype=dtype) - center
    kernel = torch.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel = kernel / (kernel.sum() + 1e-12)
    return kernel


def gaussian_blur2d(x: torch.Tensor, kernel_size: int = 11, sigma: float = None) -> torch.Tensor:
    """
    Simple Gaussian blur using depthwise separable 2D convolution.
    x: [B,C,H,W]
    kernel_size: odd integer
    sigma: if None, use a common heuristic based on kernel_size
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if sigma is None:
        # Heuristic similar to OpenCV
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    k1d = _gaussian_kernel1d(kernel_size, sigma, device, dtype)
    k2d = torch.outer(k1d, k1d)
    kernel = k2d.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(device=device, dtype=dtype)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size).contiguous()
    padding = kernel_size // 2
    # Depthwise conv
    return F.conv2d(x, kernel, bias=None, stride=1, padding=padding, groups=C)


def build_square_mask_from_scores(scores: torch.Tensor,
                                  H: int,
                                  W: int,
                                  patch_h: int,
                                  patch_w: int,
                                  ratio: float,
                                  num_squares: int = 1) -> torch.Tensor:
    """
    Place num_squares equal-sized squares to cover ~ratio of the image area, centered on top score peaks.
    Prefers non-overlapping placement; if not enough positions exist without overlap, allows overlap to
    ensure exactly num_squares squares are placed. Squares are aligned to the patch grid.
    scores: [Ph, Pw] (normalized importance scores)
    Returns a binary mask [H, W] with 1s in masked regions.
    """
    device = scores.device
    Ph, Pw = scores.shape
    total_area = int(round(ratio * H * W))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((H, W), device=device, dtype=torch.float32)

    def to_patch_aligned(side_pixels: int) -> int:
        side_pixels = max(patch_h, side_pixels)
        side_pixels = (side_pixels // patch_h) * patch_h
        return max(patch_h, min(side_pixels, min(H, W)))

    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = to_patch_aligned(side)

    mask = torch.zeros((H, W), device=device, dtype=torch.float32)
    placed = []  # list of (y0, x0, side)

    vals, idxs = torch.topk(scores.flatten(), Ph * Pw, largest=True)

    def overlaps(y0, x0, s, others):
        for (yy, xx, ss) in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

    # First pass: place up to num_squares squares without overlap
    for i in range(idxs.numel()):
        if len(placed) >= num_squares:
            break
        p = idxs[i].item()
        pr = p // Pw
        pc = p % Pw
        cy = int((pr + 0.5) * patch_h)
        cx = int((pc + 0.5) * patch_w)
        y0 = max(0, min(H - side, cy - side // 2))
        x0 = max(0, min(W - side, cx - side // 2))
        if not overlaps(y0, x0, side, placed):
            placed.append((y0, x0, side))

    # Second pass: if we still need more squares, allow overlap
    if len(placed) < num_squares:
        for i in range(idxs.numel()):
            if len(placed) >= num_squares:
                break
            p = idxs[i].item()
            pr = p // Pw
            pc = p % Pw
            cy = int((pr + 0.5) * patch_h)
            cx = int((pc + 0.5) * patch_w)
            y0 = max(0, min(H - side, cy - side // 2))
            x0 = max(0, min(W - side, cx - side // 2))
            placed.append((y0, x0, side))

    for (y0, x0, s) in placed:
        mask[y0:y0 + s, x0:x0 + s] = 1.0

    return mask


def build_random_square_mask(H: int,
                             W: int,
                             patch_h: int,
                             patch_w: int,
                             ratio: float,
                             num_squares: int = 1) -> torch.Tensor:
    """
    Place num_squares equal-sized squares to cover ~ratio of the image area at random grid-aligned positions.
    Attempts to avoid overlaps first; if not enough positions, allows overlaps to reach the desired count.
    Returns a binary mask [H, W].
    """
    device = 'cpu'
    total_area = int(round(ratio * H * W))
    if total_area <= 0 or num_squares <= 0:
        return torch.zeros((H, W), dtype=torch.float32)

    def to_patch_aligned(side_pixels: int) -> int:
        side_pixels = max(patch_h, side_pixels)
        side_pixels = (side_pixels // patch_h) * patch_h
        return max(patch_h, min(side_pixels, min(H, W)))

    side = int(round(math.sqrt(total_area / float(max(num_squares, 1)))))
    side = to_patch_aligned(side)

    max_y0 = H - side
    max_x0 = W - side
    y_positions = max(1, (max_y0 // patch_h) + 1)
    x_positions = max(1, (max_x0 // patch_w) + 1)

    mask = torch.zeros((H, W), dtype=torch.float32)
    placed = []  # list of (y0, x0, side)

    def overlaps(y0, x0, s, others):
        for (yy, xx, ss) in others:
            if not (x0 + s <= xx or xx + ss <= x0 or y0 + s <= yy or yy + ss <= y0):
                return True
        return False

    # Try to place without overlap using random sampling
    attempts = 0
    max_attempts = 1000
    while len(placed) < num_squares and attempts < max_attempts:
        ry = int(torch.randint(low=0, high=y_positions, size=(1,)).item())
        rx = int(torch.randint(low=0, high=x_positions, size=(1,)).item())
        y0 = min(max_y0, ry * patch_h)
        x0 = min(max_x0, rx * patch_w)
        if not overlaps(y0, x0, side, placed):
            placed.append((y0, x0, side))
        attempts += 1

    # If we still need more squares, allow overlaps
    while len(placed) < num_squares:
        ry = int(torch.randint(low=0, high=y_positions, size=(1,)).item())
        rx = int(torch.randint(low=0, high=x_positions, size=(1,)).item())
        y0 = min(max_y0, ry * patch_h)
        x0 = min(max_x0, rx * patch_w)
        placed.append((y0, x0, side))

    for (y0, x0, s) in placed:
        mask[y0:y0 + s, x0:x0 + s] = 1.0

    return mask

class EntREM(nn.Module):
    """
    Entropy-based REM variant: instead of masking tokens via attention, we compute a patchwise
    entropy map on the input image and build binary masks on the input.

    Masking modes:
    - Entropy mode (random_masking=False): select peaks from the entropy score map and place
      `num_squares` equal-size, grid-aligned square masks so that the union covers ~m% of the image area.
    - Random mode (random_masking=True): place `num_squares` equal-size, grid-aligned square masks
      at random positions so that the union covers ~m% of the image area.

    We then compute the REM losses across masking levels and update the model.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 steps: int = 1, episodic: bool = False,
                 m: float = 0.1, n: int = 3, lamb: float = 1.0, margin: float = 0.0,
                 patch_size: int = 16, num_bins: int = 32,
                 use_color_entropy: bool = False, entropy_weight_power: float = 2.0,
                 random_masking: bool = False,
                 num_squares: int = 1,
                 mask_type: str = 'binary',
                 prune_random_range = None,
                 # Plotting options
                 plot_loss: bool = False,
                 plot_loss_path: str = "",
                 plot_ema_alpha: float = 0.98,
                 # MCL temperature
                 mcl_temperature: float = 1.0,
                 mcl_temperature_apply: str = 'both',
                 # ERL activation selection
                 erl_activation: str = 'relu',
                 erl_leaky_relu_slope: float = 0.01,
                 erl_softplus_beta: float = 1.0,
                 # Progressive masking curriculum
                 m_step: float = 0.0,
                 m_top: float = 0.5,
                 m_progress_enable: bool = False,
                 m_progress_dir: str = 'up',
                 # Adaptive masking schedule (entropy-gap driven)
                 m_adaptive_enable: bool = False,
                 m_gap_target: float = 0.2,
                 m_gap_kp: float = 0.05,
                 m_min: float = 0.0,
                 m_max: float = 0.8,
                 m_adapt_smooth: float = 0.9,
                 # Pruning via mask-induced entropy differential
                 prune_enable: bool = False,
                 prune_tau_low: float = 0.02,
                 prune_tau_high: float = 0.5,
                 prune_mean_low: float = 0.01,
                 prune_mean_high: float = 1.0,
                 prune_skip_prediction: bool = False,
                 # Disable specific losses
                 disable_mcl: bool = False,
                 disable_erl: bool = False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EntREM requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.m = float(m)
        self.n = int(n)
        self.mn = [i * self.m for i in range(self.n)]
        # Progressive state
        self.m_progress_enable = bool(m_progress_enable)
        self.m_step = float(m_step)
        self.m_top = float(m_top)
        mpd = str(m_progress_dir).lower()
        assert mpd in ['up', 'down'], "m_progress_dir must be 'up' or 'down'"
        self.m_progress_dir = mpd
        # Current m that will be updated if progression is enabled
        self.m_current = max(0.0, min(1.0, self.m))
        # Adaptive state
        self.m_adaptive_enable = bool(m_adaptive_enable)
        self.m_gap_target = float(m_gap_target)
        self.m_gap_kp = float(m_gap_kp)
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.m_adapt_smooth = float(m_adapt_smooth)
        self._gap_ema = None
        self.lamb = lamb
        self.margin = margin

        self.entropy = Entropy()

        # Entropy-masking params
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.use_color_entropy = use_color_entropy
        self.entropy_weight_power = entropy_weight_power
        # If True, place the square mask at a random grid-aligned location instead of entropy centroid
        self.random_masking = random_masking
        # Number of equal-size squares per masking level
        self.num_squares = max(1, int(num_squares))
        # Mask fill type: 'binary' (zeros), 'mean' (per-image mean), 'gaussian' (blurred)
        mt = str(mask_type).lower()
        assert mt in ['binary', 'mean', 'gaussian'], "mask_type must be one of ['binary','mean','gaussian']"
        self.mask_type = mt

        # Plotting state
        self.plot_loss = bool(plot_loss)
        self.plot_loss_path = str(plot_loss_path) if plot_loss_path is not None else ""
        self.plot_ema_alpha = float(plot_ema_alpha)
        self._ema_mcl = None
        self._ema_erl = None
        self._steps_seen = 0
        self._ema_mcl_hist = []
        self._ema_erl_hist = []

        # MCL temperature
        self.mcl_temperature = float(mcl_temperature)
        if self.mcl_temperature <= 0:
            raise ValueError("mcl_temperature must be > 0")
        mta = str(mcl_temperature_apply).lower()
        if mta not in ['teacher', 'student', 'both']:
            raise ValueError("mcl_temperature_apply must be one of ['teacher','student','both']")
        self.mcl_temperature_apply = mta

        # ERL activation configuration
        act = str(erl_activation).lower()
        if act not in ['relu', 'leaky_relu', 'softplus', 'gelu', 'sigmoid', 'identity']:
            raise ValueError("erl_activation must be one of ['relu','leaky_relu','softplus','gelu','sigmoid','identity']")
        self.erl_activation = act
        self.erl_leaky_relu_slope = float(erl_leaky_relu_slope)
        self.erl_softplus_beta = float(erl_softplus_beta)

        # Disable flags
        self.disable_mcl = bool(disable_mcl)
        self.disable_erl = bool(disable_erl)

        # Pruning configuration
        self.prune_enable = bool(prune_enable)
        self.prune_tau_low = float(prune_tau_low)
        self.prune_tau_high = float(prune_tau_high)
        self.prune_mean_low = float(prune_mean_low)
        self.prune_mean_high = float(prune_mean_high)
        self.prune_skip_prediction = bool(prune_skip_prediction)
        # Optional dataset-level random prune range (A,B), stored for reference/logging; pruning is applied in eval scripts
        self.prune_random_range = prune_random_range
        # eval-only flag to bypass adaptation updates
        self._eval_only = False

    @contextmanager
    def no_adapt_mode(self):
        """Context manager to temporarily disable adaptation updates."""
        prev_eval_only = self._eval_only
        self._eval_only = True
        try:
            yield
        finally:
            self._eval_only = prev_eval_only

    def _current_levels(self):
        """Compute masking levels for this batch.
        If progression enabled, use m_current; else use static m.
        Levels are [0, m, 2m, ..., (n-1)m] clamped to [0,1].
        """
        m_use = self.m_current if self.m_progress_enable else self.m
        levels = [max(0.0, min(1.0, i * m_use)) for i in range(self.n)]
        # Ensure first level is exactly 0.0
        if len(levels) > 0:
            levels[0] = 0.0
        return levels

    def _step_progress(self):
        """Update m_current toward m_top by m_step in the chosen direction, with clamping.
        up: m <- min(m_top, m + m_step)
        down: m <- max(m_top, m - m_step)
        """
        if not self.m_progress_enable or self.m_step == 0.0:
            return
        if self.m_progress_dir == 'up':
            self.m_current = min(self.m_top, self.m_current + abs(self.m_step))
        else:  # down
            self.m_current = max(self.m_top, self.m_current - abs(self.m_step))
        # Clamp to valid bounds
        self.m_current = max(self.m_min, min(self.m_max, self.m_current))

    def _step_adaptive(self, gap_value: float):
        """Adapt m_current using proportional control on the entropy-gap signal.
        gap_value: observed gap across consecutive masked views (scalar float)
        m <- clamp(m + kp * (target - gap_ema), [m_min, m_max])
        """
        if not self.m_adaptive_enable:
            return
        g = float(gap_value)
        if self._gap_ema is None:
            self._gap_ema = g
        else:
            a = self.m_adapt_smooth
            self._gap_ema = a * self._gap_ema + (1.0 - a) * g
        delta = self.m_gap_kp * (self.m_gap_target - self._gap_ema)
        self.m_current = self.m_current + delta
        self.m_current = max(self.m_min, min(self.m_max, self.m_current))

    def _update_and_plot_losses(self, mcl_val: torch.Tensor, erl_val: torch.Tensor):
        if not self.plot_loss:
            return
        # Detach to CPU scalars
        mcl = float(mcl_val.detach().item())
        erl = float(erl_val.detach().item())
        alpha = self.plot_ema_alpha
        # Initialize or update EMA
        if self._ema_mcl is None:
            self._ema_mcl = mcl
            self._ema_erl = erl
        else:
            self._ema_mcl = alpha * self._ema_mcl + (1.0 - alpha) * mcl
            self._ema_erl = alpha * self._ema_erl + (1.0 - alpha) * erl
        self._steps_seen += 1
        self._ema_mcl_hist.append(self._ema_mcl)
        self._ema_erl_hist.append(self._ema_erl)

        # Guard against empty path
        if not self.plot_loss_path:
            return
        # Ensure directory exists
        out_dir = os.path.dirname(self.plot_loss_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Plot
        try:
            plt.figure(figsize=(8, 5))
            xs = list(range(1, self._steps_seen + 1))
            plt.plot(xs, self._ema_mcl_hist, label='EMA MCL', color='tab:blue')
            plt.plot(xs, self._ema_erl_hist, label='EMA ERL', color='tab:orange')
            plt.xlabel('Batch steps')
            plt.ylabel('Loss (EMA)')
            plt.title('EMA of MCL and ERL over steps')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.plot_loss_path)
            plt.close()
        except Exception:
            # Silently ignore plotting errors to avoid disrupting adaptation
            pass

    def _apply_erl_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.erl_activation == 'relu':
            return F.relu(x)
        elif self.erl_activation == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=self.erl_leaky_relu_slope)
        elif self.erl_activation == 'softplus':
            return F.softplus(x, beta=self.erl_softplus_beta)
        elif self.erl_activation == 'gelu':
            return F.gelu(x)
        elif self.erl_activation == 'sigmoid':
            return torch.sigmoid(x)
        elif self.erl_activation == 'identity':
            return x
        else:
            # Should never happen due to validation
            return F.relu(x)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    @torch.enable_grad()
    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()
        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x, self.optimizer)
        return out

    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer,
                          **kwargs) -> torch.Tensor:
        """Forward pass with multiple masked views and adaptation update."""
        # If in eval-only probe mode, bypass masking/adaptation and return base logits
        if getattr(self, "_eval_only", False):
            self.model.eval()
            with torch.no_grad():
                return self.model(x, return_attn=False)

        # Ensure size is divisible by patch_size
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input size {(H, W)} must be divisible by patch_size {self.patch_size}")
        patches_per_side = H // self.patch_size

        # Precompute entropy map only if using entropy-based masking
        ent_norm = None
        if not self.random_masking:
            xin = x
            if not self.use_color_entropy:
                xin = rgb_to_grayscale(x).clamp(0.0, 1.0)
            else:
                xin = x.clamp(0.0, 1.0)
            ent_map = compute_patch_entropy_map(xin, patches_per_side=patches_per_side, num_bins=self.num_bins)  # [B,Ph,Pw]
            # Normalize to [0,1] per image
            ent_min = ent_map.amin(dim=(1, 2), keepdim=True)
            ent_max = ent_map.amax(dim=(1, 2), keepdim=True)
            ent_norm = (ent_map - ent_min) / (ent_max - ent_min + 1e-8)

        outputs_list = []
        self.model.eval()
        levels = self._current_levels()
        for m in levels:
            if m == 0.0:
                out = self.model(x, return_attn=False)
                outputs_list.append(out)
            else:
                mfrac = m  # already fraction in [0,1]
                xb_masked = x.clone()
                Ph = patches_per_side
                Pw = patches_per_side
                patch_h = H // Ph
                patch_w = W // Pw
                # Precompute blurred fill if requested
                x_blur = None
                if self.mask_type == 'gaussian':
                    x_blur = gaussian_blur2d(xb_masked, kernel_size=11, sigma=None)
                for bi in range(B):
                    if self.random_masking:
                        mask_bw = build_random_square_mask(
                            H, W, patch_h, patch_w, ratio=mfrac, num_squares=self.num_squares
                        ).to(xb_masked.device)
                    else:
                        # Entropy-based placement using score peaks
                        mask_bw = build_square_mask_from_scores(
                            ent_norm[bi], H, W, patch_h, patch_w, ratio=mfrac, num_squares=self.num_squares
                        ).to(xb_masked.device)
                    mask_c = mask_bw.unsqueeze(0)  # [1,H,W]
                    if self.mask_type == 'binary':
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c)
                    elif self.mask_type == 'mean':
                        mean_val = xb_masked[bi].mean(dim=(1, 2), keepdim=True)  # [C,1,1]
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c) + mean_val * mask_c
                    elif self.mask_type == 'gaussian':
                        # Composite with blurred image
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
                out = self.model(xb_masked, return_attn=False)
                outputs_list.append(out)
        self.model.train()

        # Compute entropies early if pruning or adaptive schedule or ERL needed
        entropys = None
        if self.prune_enable or (not self.disable_erl) or self.m_adaptive_enable:
            entropys = [self.entropy(o) for o in outputs_list]

        # Derive keep mask based on mask-induced entropy differential statistics
        keep_mask = None
        if self.prune_enable:
            # When dataset-level random pruning is active, skip internal prune-category processing entirely
            if self.prune_random_range is not None:
                self._last_delta_mean = None
                self._last_delta_std = None
                self._last_keep_mask = None
                keep_mask = None
            else:
                if entropys is None or len(entropys) < 2:
                    # No masked levels; default to keep all
                    keep_mask = torch.ones(B, dtype=torch.bool, device=x.device)
                    # Expose empty stats for evaluators
                    self._last_delta_mean = torch.zeros(B, device=x.device).detach().cpu()
                    self._last_delta_std = torch.zeros(B, device=x.device).detach().cpu()
                    self._last_keep_mask = keep_mask.detach().cpu()
                else:
                    # Compute ΔH over masked levels relative to base (level 0)
                    deltas = []  # list of tensors [B]
                    base_h = entropys[0]
                    for i in range(1, len(entropys)):
                        deltas.append(entropys[i] - base_h)
                    delta_stack = torch.stack(deltas, dim=0)  # [K, B]
                    delta_mean = delta_stack.mean(dim=0)      # [B]
                    delta_std = delta_stack.std(dim=0, unbiased=False)  # [B]
                    # Keep if mean within [mean_low, mean_high] AND std within [tau_low, tau_high]
                    keep_by_mean = (delta_mean >= self.prune_mean_low) & (delta_mean <= self.prune_mean_high)
                    keep_by_std = (delta_std >= self.prune_tau_low) & (delta_std <= self.prune_tau_high)
                    keep_mask = keep_by_mean & keep_by_std
                    # Expose stats for evaluators
                    self._last_delta_mean = delta_mean.detach().cpu()
                    self._last_delta_std = delta_std.detach().cpu()
                    self._last_keep_mask = keep_mask.detach().cpu()
                # Guard against all-pruned edge case
                if keep_mask is not None and keep_mask.sum().item() == 0:
                    # If everything would be pruned, fall back to keeping all to avoid degenerate update
                    keep_mask = torch.ones(B, dtype=torch.bool, device=x.device)
                    self._last_keep_mask = keep_mask.detach().cpu()
        else:
            # Clear exposed stats when pruning disabled
            self._last_delta_mean = None
            self._last_delta_std = None
            self._last_keep_mask = None

        # Compute REM losses across masking levels
        mcl_loss = None
        erl_loss = None

        # Mask Consistency Loss (MCL)
        if not self.disable_mcl:
            total = 0.0
            for i in range(1, len(self.mn)):
                if self.mcl_temperature_apply == 'teacher':
                    term = softmax_entropy_temp_teacher(
                        outputs_list[i], outputs_list[0].detach(), self.mcl_temperature
                    )
                elif self.mcl_temperature_apply == 'student':
                    term = softmax_entropy_temp_student(
                        outputs_list[i], outputs_list[0].detach(), self.mcl_temperature
                    )
                else:  # both
                    term = softmax_entropy_temp(
                        outputs_list[i], outputs_list[0].detach(), self.mcl_temperature
                    )
                if keep_mask is not None:
                    if keep_mask.any():
                        total = total + term[keep_mask].mean()
                else:
                    total = total + term.mean()
                for j in range(1, i):
                    if self.mcl_temperature_apply == 'teacher':
                        term_ij = softmax_entropy_temp_teacher(
                            outputs_list[i], outputs_list[j].detach(), self.mcl_temperature
                        )
                    elif self.mcl_temperature_apply == 'student':
                        term_ij = softmax_entropy_temp_student(
                            outputs_list[i], outputs_list[j].detach(), self.mcl_temperature
                        )
                    else:
                        term_ij = softmax_entropy_temp(
                            outputs_list[i], outputs_list[j].detach(), self.mcl_temperature
                        )
                    if keep_mask is not None:
                        if keep_mask.any():
                            total = total + term_ij[keep_mask].mean()
                    else:
                        total = total + term_ij.mean()
            mcl_loss = total

        # Entropy Ranking Loss (ERL)
        if not self.disable_erl:
            margin = self.margin * math.log(outputs_list[0].shape[-1])
            total_erl = 0.0
            levels_len = len(self._current_levels())
            for i in range(levels_len):
                for j in range(i + 1, levels_len):
                    diff = entropys[i] - entropys[j].detach() + margin
                    actv = self._apply_erl_activation(diff)
                    if keep_mask is not None:
                        if keep_mask.any():
                            total_erl = total_erl + actv[keep_mask].mean()
                    else:
                        total_erl = total_erl + actv.mean()
            erl_loss = total_erl

        # Total loss and optimizer step
        loss_terms = []
        if mcl_loss is not None:
            loss_terms.append(mcl_loss)
        if erl_loss is not None:
            loss_terms.append(self.lamb * erl_loss)

        if len(loss_terms) > 0:
            loss = loss_terms[0]
            for lt in loss_terms[1:]:
                loss = loss + lt
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Update EMA trackers and save plot if enabled (zero when disabled)
        mcl_plot_val = mcl_loss if mcl_loss is not None else torch.tensor(0.0, device=x.device)
        erl_plot_val = erl_loss if erl_loss is not None else torch.tensor(0.0, device=x.device)
        self._update_and_plot_losses(mcl_val=mcl_plot_val, erl_val=erl_plot_val)

        # Compute entropy-gap and update m schedule
        if self.m_adaptive_enable and entropys is not None and len(entropys) >= 2:
            # Gap as average consecutive difference across masked levels (exclude base level step if desired)
            # Use all consecutive pairs to robustly estimate gap per step
            diffs = []
            levels = self._current_levels()
            for i in range(1, len(levels)):
                d = entropys[i] - entropys[i - 1]
                if keep_mask is not None and keep_mask.any():
                    diffs.append(d[keep_mask].mean())
                else:
                    diffs.append(d.mean())
            if len(diffs) > 0:
                gap_val = torch.stack(diffs).mean().detach().item()
                self._step_adaptive(gap_val)
        else:
            # Monotonic progression if enabled
            self._step_progress()

        # Return predictions according to pruning preference
        if self.prune_enable and keep_mask is not None:
            if self.prune_skip_prediction:
                # Return predictions for kept subset along with mask for downstream handling
                return outputs_list[0][keep_mask], keep_mask.detach()
            else:
                # Return full-batch predictions but include keep_mask for counting
                return outputs_list[0], keep_mask.detach()
        else:
            # Return full-batch predictions without mask if pruning disabled
            return outputs_list[0]
