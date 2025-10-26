from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List
from contextlib import contextmanager


class TALNLayerNorm(nn.Module):
    """LayerNorm with feature-level polynomial modulation only (no stats/affine/EMA)."""
    def __init__(self,
                 ln: nn.LayerNorm,
                 order: int = 2):
        super().__init__()
        assert isinstance(ln, nn.LayerNorm)
        self.ln = ln
        self.order = max(1, int(order))
        shape = ln.weight.shape
        # Feature-level polynomial
        self.feature_poly = FeaturePoly1D(dim=shape[0], order=self.order)
        # Current masking level
        self.current_m = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ln(x)
        self.feature_poly.current_m = float(self.current_m)
        out = self.feature_poly(out)
        return out

    def time_regularizer(self) -> torch.Tensor:
        # Regularizer removed: always return 0
        return torch.zeros((), device=self.ln.weight.device, dtype=self.ln.weight.dtype)

    def update_time_state(self):
        # No temporal state tracking required after removing stats/affine/EMA modulation
        return


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


def _wrap_layernorms_with_taln(module: nn.Module,
                               order: int):
    """Recursively replace nn.LayerNorm with TALNLayerNorm in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(
                module,
                name,
                TALNLayerNorm(
                    child,
                    order=order,
                ),
            )
        else:
            _wrap_layernorms_with_taln(child, order)


def configure_model(model: nn.Module) -> nn.Module:
    """Enable grads where needed and keep BatchNorm in special mode as in REM.

    All TALN-related wrapping has been removed.
    """
    # Move model to an appropriate device if needed
    target_device = None
    if isinstance(model, nn.DataParallel) and torch.cuda.is_available() and len(model.device_ids) > 0:
        target_device = torch.device(f"cuda:{model.device_ids[0]}")
    else:
        for p in model.parameters():
            target_device = p.device
            break
    if target_device is None:
        target_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(target_device)

    # Train-time settings for TTA
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
        elif isinstance(m, TALNLayerNorm):
            # Include base LN affine weights and biases
            if m.ln.weight.requires_grad:
                params.append(m.ln.weight)
                names.append(f"{nm}.ln.weight")
            if m.ln.bias.requires_grad:
                params.append(m.ln.bias)
                names.append(f"{nm}.ln.bias")
    return params, names


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


class FeaturePoly1D(nn.Module):
    """Learnable N-order polynomial offset conditioned on mask ratio m.
    Applies to features along the last dimension (embed dim).
    Expects inputs of shape [..., D] and adds sum_{n=1..N} kappa[n] * m^n.
    """
    def __init__(self, dim: int, order: int = 2):
        super().__init__()
        self.order = max(1, int(order))
        self.kappa = nn.Parameter(torch.zeros((self.order, dim)))
        # current masking level m in [0,1]
        self.current_m: float = 0.0

    def _delta(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        m = torch.as_tensor(self.current_m, dtype=dtype, device=device)
        powers = torch.stack([m.pow(n) for n in range(1, self.order + 1)], dim=0)  # [N]
        # [N,D] -> [D]
        return (powers.unsqueeze(1) * self.kappa).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self._delta(dtype=x.dtype, device=x.device)
        while delta.dim() < x.dim():
            delta = delta.unsqueeze(0)
        return x + delta


class PatchEmbedPolyWrapper(nn.Module):
    """Wrap a patch embedding module and apply feature polynomial after projection."""
    def __init__(self, patch_embed: nn.Module, embed_dim: int, order: int = 2):
        super().__init__()
        self.inner = patch_embed
        self.poly = FeaturePoly1D(embed_dim, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        return self.poly(out)


class AttentionPolyWrapper(nn.Module):
    """Wrap an attention module and apply feature polynomial to attention output tokens.
    Supports both signatures: inner(x)->y and inner(x)->(y, attn).
    """
    def __init__(self, attn_mod: nn.Module, embed_dim: int, order: int = 2):
        super().__init__()
        self.inner = attn_mod
        self.poly = FeaturePoly1D(embed_dim, order)

    def forward(self, x: torch.Tensor):
        out = self.inner(x)
        if isinstance(out, tuple):
            y, attn = out
            return self.poly(y), attn
        else:
            y = out
            return self.poly(y)


class MlpPolyWrapper(nn.Module):
    """Wrap an MLP module (token-wise MLP) and apply feature polynomial to its output tokens."""
    def __init__(self, mlp_mod: nn.Module, embed_dim: int, order: int = 2):
        super().__init__()
        self.inner = mlp_mod
        self.poly = FeaturePoly1D(embed_dim, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.inner(x)
        return self.poly(y)


class HeadPolyWrapper(nn.Module):
    """Wrap a classification head (nn.Linear) and apply feature polynomial to its input features."""
    def __init__(self, head_mod: nn.Module, in_dim: int, order: int = 2):
        super().__init__()
        self.inner = head_mod
        self.poly = FeaturePoly1D(in_dim, order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.poly(x)
        return self.inner(x2)


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

class SPARC(nn.Module):
    """
    Entropy-based REM variant: instead of masking tokens via attention, we compute a patchwise
    entropy map on the input image and build binary masks on the input.

    Masking modes:
    - Random mode (random_masking=True): place `num_squares` equal-size, grid-aligned square masks
      at random positions so that the union covers ~m% of the image area.

    We then compute the REM losses across masking levels and update the model.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 steps: int = 1, episodic: bool = False,
                 m: float = 0.1, n: int = 3, lamb: float = 1.0, margin: float = 0.0,
                 patch_size: int = 16,
                 random_masking: bool = True,
                 num_squares: int = 1,
                 mask_type: str = 'binary',
                 # Plotting options
                 plot_loss: bool = False,
                 plot_loss_path: str = "",
                 plot_ema_alpha: float = 0.98,
                 # MCL temperature
                 mcl_temperature: float = 1.0,
                 mcl_temperature_apply: str = 'both',
                 mcl_distance: str = 'ce',
                 # ERL activation selection
                 erl_activation: str = 'relu',
                 erl_leaky_relu_slope: float = 0.01,
                 erl_softplus_beta: float = 1.0,
                 # (Removed progressive/adaptive masking and internal pruning controls)
                 # Disable specific losses
                 disable_mcl: bool = False,
                 disable_erl: bool = False,
                 # Logsparc options
                 logsparc_enable: str = 'none',
                 logsparc_lr_mult: float = 1.0,
                 logsparc_reg: float = 0.0,
                 logsparc_temp: float = 0.0,
                 logsparc_type2: bool = False,
                 logsparc_type3: bool = False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "SPARC requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.m = float(m)
        self.n = int(n)
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = lamb
        self.margin = margin

        self.entropy = Entropy()

        # Entropy-masking params (fixed settings)
        self.patch_size = patch_size
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
        self._ema_mcl_hist = []
        self._ema_erl_hist = []
        self._steps_seen = 0

        # MCL temperature
        self.mcl_temperature = float(mcl_temperature)
        if self.mcl_temperature <= 0:
            raise ValueError("mcl_temperature must be > 0")
        mta = str(mcl_temperature_apply).lower()
        if mta not in ['teacher', 'student', 'both']:
            raise ValueError("mcl_temperature_apply must be one of ['teacher','student','both']")
        self.mcl_temperature_apply = mta
        # MCL distance metric
        mdist = str(mcl_distance).lower()
        if mdist not in ['ce', 'kl', 'js', 'mse', 'mae']:
            raise ValueError("mcl_distance must be one of ['ce','kl','js','mse','mae']")
        self.mcl_distance = mdist

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
        # eval-only flag to bypass adaptation updates
        self._eval_only = False

        # Logsparc state
        self.logsparc_enable = str(logsparc_enable).lower()
        if self.logsparc_enable not in ['none', 'gamma', 'beta', 'gammabeta']:
            self.logsparc_enable = 'none'
        self.logsparc_lr_mult = float(logsparc_lr_mult)
        self.logsparc_reg = float(logsparc_reg)
        self.logsparc_head: nn.Linear = None  # lazy init when CLS/logits dim known
        self._logsparc_params_added = False
        self.logsparc_temp = float(logsparc_temp)
        self.logsparc_type2 = bool(logsparc_type2)
        self.logsparc_type3 = bool(logsparc_type3)

        # Debug logging control
        self._debug_poly_log_limit = 3
        self._debug_poly_log_count = 0

    @contextmanager
    def no_adapt_mode(self):
        """Context manager to temporarily disable adaptation updates."""
        prev_eval_only = self._eval_only
        self._eval_only = True
        try:
            yield
        finally:
            self._eval_only = prev_eval_only

    def _get_cls_and_logits(self, x: torch.Tensor):
        """Try to extract class token feature [B,D] and logits [B,K] in one pass when possible.
        Falls back to calling model(x) for logits and returns (None, logits) if unsupported.
        """
        base = self.model.module if hasattr(self.model, 'module') else self.model
        cls = None
        logits = None
        try:
            if hasattr(base, 'forward_features'):
                # Many ViTs support this
                feats = base.forward_features(x)
                if isinstance(feats, tuple):
                    cls = feats[0]
                else:
                    cls = feats
                if hasattr(base, 'forward_head'):
                    logits = base.forward_head(cls)
                elif hasattr(base, 'head') and isinstance(base.head, nn.Module):
                    logits = base.head(cls)
                else:
                    # Can't map to logits, fallback
                    cls = None
                    logits = self.model(x, return_attn=False)
            else:
                logits = self.model(x, return_attn=False)
        except Exception:
            # Last resort
            logits = self.model(x, return_attn=False)
            cls = None
        return cls, logits

    def _current_levels(self):
        """Compute masking levels for this batch.
        Levels are [0, m, 2m, ..., (n-1)m] clamped to [0,1] with a static m.
        """
        levels = [max(0.0, min(1.0, i * self.m)) for i in range(self.n)]
        if len(levels) > 0:
            levels[0] = 0.0
        return levels

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

    def _mcl_pair_distance(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-sample distance between student and teacher distributions according
        to the configured MCL distance and temperature application.

        Returns a tensor [B].
        """
        dist = self.mcl_distance
        apply = self.mcl_temperature_apply
        t = self.mcl_temperature

        if dist == 'ce':
            # Use existing TorchScript CE helpers for efficiency and parity
            if apply == 'teacher':
                return softmax_entropy_temp_teacher(student_logits, teacher_logits.detach(), t)
            elif apply == 'student':
                return softmax_entropy_temp_student(student_logits, teacher_logits.detach(), t)
            else:  # both
                return softmax_entropy_temp(student_logits, teacher_logits.detach(), t)

        # Prepare temperature-adjusted logits for other distances
        s = student_logits
        te = teacher_logits.detach()
        if apply == 'teacher':
            te_t = te / t
            s_t = s
        elif apply == 'student':
            te_t = te
            s_t = s / t
        else:  # both
            te_t = te / t
            s_t = s / t

        # Probabilities and log-probabilities as needed
        p = te_t.softmax(dim=1)            # teacher distribution
        logp = te_t.log_softmax(dim=1)
        q = s_t.softmax(dim=1)             # student distribution
        logq = s_t.log_softmax(dim=1)

        if dist == 'kl':
            # KL(p || q)
            kl = (p * (logp - logq)).sum(dim=1)
            return kl
        elif dist == 'js':
            # JS(p || q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), m=(p+q)/2
            m = 0.5 * (p + q)
            eps = 1e-8
            logm = (m + eps).log()
            kl_p_m = (p * (logp - logm)).sum(dim=1)
            kl_q_m = (q * (logq - logm)).sum(dim=1)
            return 0.5 * (kl_p_m + kl_q_m)
        elif dist == 'mse':
            # Mean squared error between distributions
            mse = ((p - q) ** 2).mean(dim=1)
            return mse
        elif dist == 'mae':
            # Mean absolute error between distributions
            mae = (p - q).abs().mean(dim=1)
            return mae
        else:
            # Fallback to CE (should not happen)
            return softmax_entropy_temp(student_logits, teacher_logits.detach(), t)

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

        # other masking than random masking is not supported
        ent_norm = None
        if not self.random_masking:
            raise ValueError("other masking than random masking is not supported")

        outputs_list = []
        # For Logsparc regularizer: store gamma/beta per level (including m=0 for reg only)
        logsparc_gamma_levels: List[torch.Tensor] = []
        logsparc_beta_levels: List[torch.Tensor] = []
        self.model.eval()
        levels = self._current_levels()
        for m in levels:
            if m == 0.0:
                # Compute logits normally; also get class token for regularizer only
                cls0, out0 = self._get_cls_and_logits(x)
                if isinstance(out0, tuple):
                    out0 = out0[0]
                outputs_list.append(out0)
                # For regularizer, compute gamma/beta at m=0 (not applied to logits)
                if (self.logsparc_enable != 'none') and (self.logsparc_reg > 0.0) and (cls0 is not None):
                    # Lazy-create Logsparc head
                    if self.logsparc_head is None:
                        in_dim = int(cls0.shape[-1])
                        K = int(out0.shape[-1])
                        out_dim = 2 if (not self.logsparc_type2 and not self.logsparc_type3) else (2 * K)
                        self.logsparc_head = nn.Linear(in_dim, out_dim).to(cls0.device)
                        # Add params with LR multiplier
                        if not self._logsparc_params_added:
                            try:
                                base_lr = self.optimizer.param_groups[0].get('lr', None)
                                if base_lr is None:
                                    self.optimizer.add_param_group({'params': self.logsparc_head.parameters()})
                                else:
                                    self.optimizer.add_param_group({'params': self.logsparc_head.parameters(), 'lr': base_lr * self.logsparc_lr_mult})
                            except Exception:
                                pass
                            self._logsparc_params_added = True
                    # Compute gamma/beta for regularizer only (do not apply at m=0)
                    if self.logsparc_type3:
                        # global per-class
                        cls_avg = cls0.mean(dim=0, keepdim=True)  # [1,D]
                        raw = self.logsparc_head(cls_avg)  # [1,2*K]
                        K = raw.shape[-1] // 2
                        raw_g = raw[:, :K]
                        raw_b = raw[:, K:]
                        # No temperature at m=0 (regularizer only)
                        gb_g = F.softplus(raw_g)
                        gb_b = F.softplus(raw_b)
                        g0 = gb_g.squeeze(0)  # [K]
                        b0 = gb_b.squeeze(0)  # [K]
                    elif self.logsparc_type2:
                        # per-class per-sample
                        raw = self.logsparc_head(cls0)  # [B,2*K]
                        K = raw.shape[-1] // 2
                        raw_g = raw[:, :K]
                        raw_b = raw[:, K:]
                        gb_g = F.softplus(raw_g)
                        gb_b = F.softplus(raw_b)
                        g0 = gb_g  # [B,K]
                        b0 = gb_b  # [B,K]
                    else:
                        raw = self.logsparc_head(cls0)  # [B,2]
                        raw_g = raw[:, 0]
                        raw_b = raw[:, 1]
                        gb_g = F.softplus(raw_g)
                        gb_b = F.softplus(raw_b)
                        g0 = gb_g   # [B]
                        b0 = gb_b   # [B]
                    logsparc_gamma_levels.append(g0)
                    logsparc_beta_levels.append(b0)
                else:
                    logsparc_gamma_levels.append(None)
                    logsparc_beta_levels.append(None)
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
                        raise ValueError("other masking than random masking is not supported")
                    mask_c = mask_bw.unsqueeze(0)  # [1,H,W]
                    if self.mask_type == 'binary':
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c)
                    elif self.mask_type == 'mean':
                        mean_val = xb_masked[bi].mean(dim=(1, 2), keepdim=True)  # [C,1,1]
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c) + mean_val * mask_c
                    elif self.mask_type == 'gaussian':
                        # Composite with blurred image
                        xb_masked[bi] = xb_masked[bi] * (1.0 - mask_c) + x_blur[bi] * mask_c
                        # Update modules with current masking level before forward
                for mod in self.model.modules():
                    if hasattr(mod, 'current_m'):
                        mod.current_m = float(mfrac)

                # Debug: on first few batches, log feature-poly delta norms
                if self._debug_poly_log_count < self._debug_poly_log_limit:
                    try:
                        self._debug_log_feature_poly(mfrac, xb_masked)
                        self._debug_poly_log_count += 1
                    except Exception:
                        pass
                # Compute class token and logits for masked input
                cls_m, out_m = self._get_cls_and_logits(xb_masked)
                if isinstance(out_m, tuple):
                    out_m = out_m[0]
                # Apply Logsparc only for masked images if enabled and class token available
                if (self.logsparc_enable != 'none') and (cls_m is not None):
                    # Lazy-create Logsparc head
                    if self.logsparc_head is None:
                        in_dim = int(cls_m.shape[-1])
                        K = int(out_m.shape[-1])
                        out_dim = 2 if (not self.logsparc_type2 and not self.logsparc_type3) else (2 * K)
                        self.logsparc_head = nn.Linear(in_dim, out_dim).to(cls_m.device)
                        if not self._logsparc_params_added:
                            try:
                                base_lr = self.optimizer.param_groups[0].get('lr', None)
                                if base_lr is None:
                                    self.optimizer.add_param_group({'params': self.logsparc_head.parameters()})
                                else:
                                    self.optimizer.add_param_group({'params': self.logsparc_head.parameters(), 'lr': base_lr * self.logsparc_lr_mult})
                            except Exception:
                                pass
                            self._logsparc_params_added = True
                    # Compute gamma/beta depending on type
                    if self.logsparc_type3:
                        # global per-class, use batch-average cls
                        cls_avg = cls_m.mean(dim=0, keepdim=True)  # [1,D]
                        raw = self.logsparc_head(cls_avg)  # [1,2*K]
                        K = raw.shape[-1] // 2
                        raw_g = raw[:, :K]
                        raw_b = raw[:, K:]
                        # Apply temperature to beta only for masked views
                        if self.logsparc_temp > 0.0:
                            raw_b = raw_b / self.logsparc_temp
                        gb_g = F.softplus(raw_g)
                        gb_b = F.softplus(raw_b)
                        gamma_b = gb_g   # [1,K]
                        beta_b = gb_b    # [1,K]
                        # expand to batch when applying
                        gamma_apply = gamma_b.expand(out_m.shape[0], -1)  # [B,K]
                        beta_apply = beta_b.expand(out_m.shape[0], -1)    # [B,K]
                    elif self.logsparc_type2:
                        # per-class per-sample
                        raw = self.logsparc_head(cls_m)  # [B,2*K]
                        K = raw.shape[-1] // 2
                        raw_g = raw[:, :K]
                        raw_b = raw[:, K:]
                        if self.logsparc_temp > 0.0:
                            raw_b = raw_b / self.logsparc_temp
                        gb_g = F.softplus(raw_g)
                        gb_b = F.softplus(raw_b)
                        gamma_apply = gb_g  # [B,K]
                        beta_apply = gb_b   # [B,K]
                        gamma_b = gamma_apply
                        beta_b = beta_apply
                    else:
                        # scalar per-sample
                        raw = self.logsparc_head(cls_m)  # [B,2]
                        raw_g = raw[:, 0]
                        raw_b = raw[:, 1]
                        if self.logsparc_temp > 0.0:
                            raw_b = raw_b / self.logsparc_temp
                        gb_g = F.softplus(raw_g)  # non-negative
                        gb_b = F.softplus(raw_b)
                        gamma_apply = gb_g.unsqueeze(1)  # [B,1]
                        beta_apply = gb_b.unsqueeze(1)   # [B,1]
                        gamma_b = gamma_apply.squeeze(1)  # [B]
                        beta_b = beta_apply.squeeze(1)   # [B]
                    # Mode selection
                    if self.logsparc_enable == 'gamma':
                        beta_use = torch.zeros_like(beta_apply)
                        gamma_use = gamma_apply
                    elif self.logsparc_enable == 'beta':
                        beta_use = beta_apply
                        gamma_use = torch.ones_like(beta_apply)
                    else:  # 'gammabeta'
                        beta_use = beta_apply
                        gamma_use = gamma_apply
                    xform = out_m * gamma_use + beta_use
                    # Always L2-normalize transformed logits (temperature now only affects beta pre-softplus)
                    eps = 1e-6
                    mag = torch.norm(xform, p=2, dim=1, keepdim=True).clamp_min(eps)
                    out_m = xform / mag
                    # Save gamma/beta for regularizer
                    logsparc_gamma_levels.append(gamma_b)
                    logsparc_beta_levels.append(beta_b)
                else:
                    logsparc_gamma_levels.append(None)
                    logsparc_beta_levels.append(None)
                outputs_list.append(out_m)
        self.model.train()

        # Losses computed on raw outputs (MARN removed)
        outputs_for_losses = outputs_list
        levels_for_losses = levels

        # Compute entropies early if ERL needed
        entropys = None
        if not self.disable_erl:
            entropys = [self.entropy(o) for o in outputs_list]

        mcl_loss = None
        erl_loss = None

        # Mask Consistency Loss (MCL)
        if not self.disable_mcl:
            total_mcl = None
            # Need at least 2 levels for any pairwise distance
            for i in range(1, len(self.mn)):
                term = self._mcl_pair_distance(outputs_for_losses[i], outputs_for_losses[0]).mean()
                total_mcl = term if total_mcl is None else (total_mcl + term)
                for j in range(1, i):
                    term_ij = self._mcl_pair_distance(outputs_for_losses[i], outputs_for_losses[j]).mean()
                    total_mcl = term_ij if total_mcl is None else (total_mcl + term_ij)
            mcl_loss = total_mcl  # may remain None if no pairs

        # Entropy Ranking Loss (ERL)
        if not self.disable_erl:
            margin = self.margin * math.log(outputs_list[0].shape[-1])
            total_erl = None
            levels_len = len(self._current_levels())
            for i in range(levels_len):
                for j in range(i + 1, levels_len):
                    ent_i = entropys[i]
                    ent_j = entropys[j].detach()
                    diff = ent_i - ent_j + margin
                    actv = self._apply_erl_activation(diff).mean()
                    total_erl = actv if total_erl is None else (total_erl + actv)
            erl_loss = total_erl  # may remain None if no pairs

        # TALN temporal regularizer removed (no-op)
        taln_reg = None

        # Logsparc monotonic regularizer across levels (including m=0 baseline if available)
        logsparc_reg_loss = None
        if (self.logsparc_enable != 'none') and (self.logsparc_reg > 0.0):
            try:
                # Build tensors for consecutive level penalties where gamma/beta exist
                penalties = []
                for seq in (logsparc_gamma_levels, logsparc_beta_levels):
                    prev = None
                    for val in seq:
                        if val is None:
                            prev = None
                            continue
                        if prev is None:
                            prev = val
                            continue
                        # Enforce non-decreasing: penalize decreases
                        penalties.append(F.relu(prev - val).mean())
                        prev = val
                if len(penalties) > 0:
                    logsparc_reg_loss = sum(penalties) / float(len(penalties))
            except Exception:
                logsparc_reg_loss = None

        # Total loss and optimizer step
        loss_terms = []
        if isinstance(mcl_loss, torch.Tensor) and mcl_loss.requires_grad:
            loss_terms.append(mcl_loss)
        if isinstance(erl_loss, torch.Tensor) and erl_loss.requires_grad:
            loss_terms.append(self.lamb * erl_loss)
        # taln_reg is a no-op and not included

        if (logsparc_reg_loss is not None) and (self.logsparc_reg > 0.0):
            loss_terms.append(self.logsparc_reg * logsparc_reg_loss)
        if len(loss_terms) > 0:
            loss = loss_terms[0]
            for lt in loss_terms[1:]:
                loss = loss + lt
            loss.backward()
            # Debug: on first few batches, log grad norms for feature-polys
            if self._debug_poly_log_count < self._debug_poly_log_limit:
                try:
                    self._debug_log_feature_poly_grads()
                except Exception:
                    pass
            optimizer.step()
            # After update, advance TALN temporal buffers
            for mod in self.model.modules():
                if isinstance(mod, TALNLayerNorm):
                    mod.update_time_state()
            optimizer.zero_grad()

        # Update EMA trackers and save plot if enabled (zero when disabled)
        mcl_plot_val = mcl_loss if mcl_loss is not None else torch.tensor(0.0, device=x.device)
        erl_plot_val = erl_loss if erl_loss is not None else torch.tensor(0.0, device=x.device)
        self._update_and_plot_losses(mcl_val=mcl_plot_val, erl_val=erl_plot_val)

        # Return full-batch predictions
        return outputs_list[0]

    def _debug_log_feature_poly(self, mfrac: float, xb: torch.Tensor):
        logger = logging.getLogger(__name__)
        # unwrap model for access
        base = self.model.module if hasattr(self.model, 'module') else self.model
        device = xb.device
        dtype = xb.dtype
        msgs = [f"[SPARC.debug] m={mfrac:.3f}"]
        # Patch embed
        if hasattr(base, 'patch_embed') and isinstance(base.patch_embed, PatchEmbedPolyWrapper):
            d = base.patch_embed.poly._delta(dtype=dtype, device=device)
            msgs.append(f"patch_poly|kappa_mean={base.patch_embed.poly.kappa.abs().mean().item():.4e}, delta_mean={d.abs().mean().item():.4e}")
        # Attention blocks
        attn_deltas = []
        if hasattr(base, 'blocks'):
            for i, blk in enumerate(base.blocks):
                if hasattr(blk, 'attn') and isinstance(blk.attn, AttentionPolyWrapper):
                    d = blk.attn.poly._delta(dtype=dtype, device=device)
                    attn_deltas.append(d.abs().mean().item())
        if attn_deltas:
            msgs.append(f"attn_poly|blocks={len(attn_deltas)}, delta_mean_avg={float(sum(attn_deltas)/len(attn_deltas)):.4e}")
        # Class token poly
        if hasattr(base, 'taln_cls_poly') and isinstance(base.taln_cls_poly, FeaturePoly1D):
            d = base.taln_cls_poly._delta(dtype=dtype, device=device)
            msgs.append(f"cls_poly|kappa_mean={base.taln_cls_poly.kappa.abs().mean().item():.4e}, delta_mean={d.abs().mean().item():.4e}")
        logger.info(" | ".join(msgs))

    def _debug_log_feature_poly_grads(self):
        logger = logging.getLogger(__name__)
        base = self.model.module if hasattr(self.model, 'module') else self.model
        msgs = ["[SPARC.debug] grad"]
        # Patch embed
        if hasattr(base, 'patch_embed') and isinstance(base.patch_embed, PatchEmbedPolyWrapper):
            g = base.patch_embed.poly.kappa.grad
            gnorm = float(g.norm().item()) if (g is not None) else float('nan')
            msgs.append(f"patch_poly|grad_norm={gnorm:.4e}")
        # Attention blocks
        attn_norms = []
        if hasattr(base, 'blocks'):
            for blk in base.blocks:
                if hasattr(blk, 'attn') and isinstance(blk.attn, AttentionPolyWrapper):
                    g = blk.attn.poly.kappa.grad
                    if g is not None:
                        attn_norms.append(float(g.norm().item()))
        if attn_norms:
            avg = sum(attn_norms) / len(attn_norms)
            msgs.append(f"attn_poly|blocks={len(attn_norms)}, grad_norm_avg={avg:.4e}")
        # Class token poly
        if hasattr(base, 'taln_cls_poly') and isinstance(base.taln_cls_poly, FeaturePoly1D):
            g = base.taln_cls_poly.kappa.grad
            gnorm = float(g.norm().item()) if (g is not None) else float('nan')
            msgs.append(f"cls_poly|grad_norm={gnorm:.4e}")
        logger.info(" | ".join(msgs))
