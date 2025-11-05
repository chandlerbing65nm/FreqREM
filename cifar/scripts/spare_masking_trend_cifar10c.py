#!/usr/bin/env python3
import os
import sys
import math
import argparse
from collections import OrderedDict
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

# Ensure local robustbench fork is importable when running from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CIFAR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if CIFAR_DIR not in sys.path:
    sys.path.insert(0, CIFAR_DIR)

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


# CIFAR-10 class names for labeling saved figures
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # Same as cifar/rem.py Entropy:  -(softmax * log_softmax).sum(1)
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def build_source_model(ckpt_dir: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    # Use REM-capable ViT so we can request attention and masking via len_keep (not used here)
    arch = 'Standard_VITB_REM'
    model = load_model(arch, ckpt_dir, 'cifar10', ThreatModel.corruptions)
    # Load local checkpoint (strip potential 'module.' prefix)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = rm_substr_from_state_dict(state, 'module.')
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


# -------- Frequency masking utilities --------

def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))

def _ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))


def precompute_radius_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Returns a [H, W] tensor with Euclidean distance from the centered origin (after fftshift).
    """
    yy = torch.arange(H, device=device).float() - (H // 2)
    xx = torch.arange(W, device=device).float() - (W // 2)
    Y, X = torch.meshgrid(yy, xx, indexing='ij')
    R = torch.sqrt(Y * Y + X * X)
    return R


def apply_frequency_mask(x: torch.Tensor,
                         mask_percent: float,
                         mode: str = 'lowest_energy') -> torch.Tensor:
    """
    Apply a frequency-domain mask while always preserving the DC component.

    Modes:
    - 'lowest_energy': progressively zeros the lowest-energy frequency bins per image
      (adaptive per image).
    - 'radial_highfreq': progressively zeros the highest-radius frequency bins
      (same mask for all images in the batch). This behaves like a low-pass
      filter and yields visually increasing blur as the percentage grows.
    - 'radial_lowfreq': progressively zeros the lowest-radius frequency bins
      (same mask for all images). This behaves like a high-pass filter, removing
      low frequencies around DC (DC itself is preserved).

    x: [B,C,H,W] in [0,1]
    mask_percent: percentage (0-100) of non-DC frequency bins to zero.
    Returns masked image [B,C,H,W] in [0,1].
    """
    assert 0.0 <= mask_percent <= 100.0
    B, C, H, W = x.shape

    # Forward FFT (complex). No shift needed; DC is at index (0,0).
    X = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1), norm='ortho')  # [B,C,H,W]

    # Number of bins to mask (random selection across spatial frequencies; applies to all channels)
    total_bins = H * W
    k = int(math.ceil((mask_percent / 100.0) * total_bins))
    if k <= 0:
        X_masked = X
    else:
        # Build per-image random masks. Mask is shared across channels per image.
        mask = torch.ones((B, 1, H, W), device=x.device, dtype=X.dtype)
        flat_mask = mask.view(B, -1)
        for i in range(B):
            idx = torch.randperm(total_bins, device=x.device)[:k]
            flat_mask[i, idx] = 0
        mask = flat_mask.view(B, 1, H, W)
        X_masked = X * mask

    # Inverse FFT back to image space
    x_rec = torch.fft.ifft2(X_masked, dim=(-2, -1), norm='ortho').real
    x_rec = x_rec.clamp(0.0, 1.0)
    return x_rec


# -------- Spatial masking utilities (ported) --------

def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b)


def compute_patch_entropy_map(x_img: torch.Tensor,
                              patches_per_side: int = 24,
                              num_bins: int = 16) -> torch.Tensor:
    B, C, H, W = x_img.shape
    patch_h = H // patches_per_side
    patch_w = W // patches_per_side
    assert H % patches_per_side == 0 and W % patches_per_side == 0, "H and W must be divisible by patches_per_side"
    patches = x_img.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    Ph = patches.size(2)
    Pw = patches.size(3)
    patch_size = patch_h * patch_w
    patches = patches.contiguous().view(B, C, Ph, Pw, patch_size)
    patches_flat = patches.view(B * C * Ph * Pw, patch_size)
    q = torch.clamp((patches_flat * float(num_bins)).long(), 0, num_bins - 1)
    ones = torch.ones_like(q, dtype=torch.float32)
    counts = torch.zeros(q.size(0), num_bins, device=x_img.device, dtype=torch.float32)
    counts.scatter_add_(1, q, ones)
    probs = counts / float(patch_size)
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    ent = ent.view(B, C, Ph, Pw)
    ent = ent.mean(dim=1)
    return ent


def compute_image_spatial_entropy(x_img: torch.Tensor, num_bins: int = 64) -> torch.Tensor:
    B, C, H, W = x_img.shape
    xin = x_img if C == 1 else rgb_to_grayscale(x_img)
    xin = xin.clamp(0.0, 1.0)
    N = H * W
    q = torch.clamp((xin.view(B, -1) * float(num_bins)).long(), 0, num_bins - 1)
    ones = torch.ones_like(q, dtype=torch.float32)
    counts = torch.zeros((B, num_bins), device=x_img.device, dtype=torch.float32)
    counts.scatter_add_(1, q, ones)
    probs = counts / float(N)
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    return ent


def compute_mask_frequency_energy(x_img: torch.Tensor, mask_bw: torch.Tensor, use_color: bool = False) -> torch.Tensor:
    B, C, H, W = x_img.shape
    if not use_color:
        xg = rgb_to_grayscale(x_img) if C == 3 else x_img
    else:
        xg = x_img
    x_sel = xg * mask_bw.unsqueeze(1)
    Xf = torch.fft.fftn(x_sel, dim=(-2, -1), norm='ortho')
    power = (Xf.real**2 + Xf.imag**2)
    power_sum = power.sum(dim=(-2, -1))
    if not use_color:
        out = power_sum[:, 0]
    else:
        out = power_sum.mean(dim=1)
    return out


def build_centered_square_mask(H: int,
                               W: int,
                               side: int,
                               cy: int,
                               cx: int) -> torch.Tensor:
    y0 = max(0, min(H - side, cy - side // 2))
    x0 = max(0, min(W - side, cx - side // 2))
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[y0:y0 + side, x0:x0 + side] = 1.0
    return mask


def evaluate_entropy_masking_trend(model: torch.nn.Module,
                                   x: torch.Tensor,
                                   y: torch.Tensor,
                                   device: torch.device,
                                   batch_size: int = 50,
                                   ratios: List[int] = None,
                                   save_mask_examples: int = 0,
                                   mask_example_levels: List[int] = None,
                                   mask_figs_dir: str = None,
                                   example_tag: str = "",
                                   patches_per_side: int = 24,
                                   num_bins: int = 16,
                                   spatial_entropy_bins: int = 64,
                                   use_color_entropy: bool = False,
                                   entropy_weight_power: float = 2.0,
                                   target_class_idx: Optional[int] = None,
                                   masking_mode: str = 'entropy',
                                   rng: Optional[torch.Generator] = None) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    model.eval()
    if ratios is None:
        ratios = [i for i in range(0, 101, 10)]
    total = 0
    correct_per_ratio = {r: 0 for r in ratios}
    entropy_sum_per_ratio = {r: 0.0 for r in ratios}
    spatial_entropy_sum_per_ratio = {r: 0.0 for r in ratios}
    freq_energy_sum_per_ratio = {r: 0.0 for r in ratios}
    if mask_example_levels is None:
        mask_example_levels = [0, 10, 20]
    saved_examples = 0
    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)
            B, C, H, W = xb.shape
            patch_h = H // patches_per_side
            patch_w = W // patches_per_side
            xin = xb if use_color_entropy else rgb_to_grayscale(xb)
            xin = xin.clamp(0.0, 1.0)
            ent_map = compute_patch_entropy_map(xin, patches_per_side=patches_per_side, num_bins=num_bins)
            ent_min = ent_map.amin(dim=(1,2), keepdim=True)
            ent_max = ent_map.amax(dim=(1,2), keepdim=True)
            ent_norm = (ent_map - ent_min) / (ent_max - ent_min + 1e-8)
            for r in ratios:
                m = r / 100.0
                xb_masked = xb.clone()
                mask_batch = torch.zeros((B, H, W), dtype=torch.float32, device=xb.device)
                if m != 0.0:
                    if masking_mode == 'entropy':
                        Ph, Pw = ent_norm.shape[1], ent_norm.shape[2]
                        Np = Ph * Pw
                        k = max(1, int(round(m * Np)))
                        for bi in range(B):
                            scores_b = ent_norm[bi].flatten()
                            vals, idxs = torch.topk(scores_b, k, largest=True)
                            rows = (idxs // Pw).float()
                            cols = (idxs % Pw).float()
                            eps_w = 1e-8
                            w = (vals.float() + eps_w) ** float(entropy_weight_power)
                            r_bar = (rows * w).sum() / w.sum()
                            c_bar = (cols * w).sum() / w.sum()
                            cy = int(torch.round((r_bar + 0.5) * patch_h).item())
                            cx = int(torch.round((c_bar + 0.5) * patch_w).item())
                            total_area = int(round(m * H * W))
                            side = int(round(math.sqrt(max(total_area, 1))))
                            side = max(patch_h, (side // patch_h) * patch_h)
                            side = min(side, min(H, W))
                            mask_bw = build_centered_square_mask(H, W, side, cy, cx).to(xb_masked.device)
                            mask_batch[bi] = mask_bw
                            xb_masked[bi] = xb_masked[bi] * (1.0 - mask_bw.unsqueeze(0))
                    elif masking_mode == 'random':
                        total_area = int(round(m * H * W))
                        side = int(round(math.sqrt(max(total_area, 1))))
                        side = max(patch_h, (side // patch_h) * patch_h)
                        side = min(side, min(H, W))
                        high_y = max(H - side + 1, 1)
                        high_x = max(W - side + 1, 1)
                        for bi in range(B):
                            if rng is not None:
                                y0 = int(torch.randint(low=0, high=high_y, size=(1,), generator=rng).item())
                                x0 = int(torch.randint(low=0, high=high_x, size=(1,), generator=rng).item())
                            else:
                                y0 = int(torch.randint(low=0, high=high_y, size=(1,)).item())
                                x0 = int(torch.randint(low=0, high=high_x, size=(1,)).item())
                            mask_bw = torch.zeros((H, W), dtype=torch.float32, device=xb_masked.device)
                            mask_bw[y0:y0 + side, x0:x0 + side] = 1.0
                            mask_batch[bi] = mask_bw
                            xb_masked[bi] = xb_masked[bi] * (1.0 - mask_bw.unsqueeze(0))
                    else:
                        raise ValueError(f"Unknown masking_mode '{masking_mode}'. Use 'entropy' or 'random'.")
                sp_ent_b = compute_image_spatial_entropy(xb_masked, num_bins=spatial_entropy_bins)
                spatial_entropy_sum_per_ratio[r] += float(sp_ent_b.sum().item())
                freq_b = compute_mask_frequency_energy(xb, mask_batch, use_color=False)
                freq_energy_sum_per_ratio[r] += float(freq_b.sum().item())
                logits = model(xb_masked, return_attn=False)
                pred = logits.argmax(dim=1)
                correct = (pred == yb).sum().item()
                ent = entropy_from_logits(logits).sum().item()
                correct_per_ratio[r] += correct
                entropy_sum_per_ratio[r] += ent
            if save_mask_examples > 0 and saved_examples < save_mask_examples and mask_figs_dir is not None:
                os.makedirs(mask_figs_dir, exist_ok=True)
                for bi in range(B):
                    if saved_examples >= save_mask_examples:
                        break
                    if target_class_idx is not None and int(yb[bi].detach().cpu().item()) != target_class_idx:
                        continue
                    imgs = []
                    labels = []
                    attn_maps = []
                    global_idx = start + bi
                    class_idx = int(yb[bi].detach().cpu().item())
                    class_name = CIFAR10_CLASSES[class_idx] if 0 <= class_idx < len(CIFAR10_CLASSES) else f"class{class_idx}"
                    for lv in mask_example_levels:
                        m = float(lv) / 100.0
                        if m == 0.0:
                            mask_bw = torch.zeros((H, W), dtype=torch.float32)
                        else:
                            if masking_mode == 'entropy':
                                Ph, Pw = ent_norm.shape[1], ent_norm.shape[2]
                                Np = Ph * Pw
                                k = max(1, int(round(m * Np)))
                                scores_b = ent_norm[bi].flatten()
                                vals, idxs = torch.topk(scores_b, k, largest=True)
                                rows = (idxs // Pw).float()
                                cols = (idxs % Pw).float()
                                eps_w = 1e-8
                                w = (vals.float() + eps_w) ** float(entropy_weight_power)
                                r_bar = (rows * w).sum() / w.sum()
                                c_bar = (cols * w).sum() / w.sum()
                                cy = int(torch.round((r_bar + 0.5) * patch_h).item())
                                cx = int(torch.round((c_bar + 0.5) * patch_w).item())
                                total_area = int(round(m * H * W))
                                side = int(round(math.sqrt(max(total_area, 1))))
                                side = max(patch_h, (side // patch_h) * patch_h)
                                side = min(side, min(H, W))
                                mask_bw = build_centered_square_mask(H, W, side, cy, cx)
                            elif masking_mode == 'random':
                                total_area = int(round(m * H * W))
                                side = int(round(math.sqrt(max(total_area, 1))))
                                side = max(patch_h, (side // patch_h) * patch_h)
                                side = min(side, min(H, W))
                                high_y = max(H - side + 1, 1)
                                high_x = max(W - side + 1, 1)
                                if rng is not None:
                                    y0 = int(torch.randint(low=0, high=high_y, size=(1,), generator=rng).item())
                                    x0 = int(torch.randint(low=0, high=high_x, size=(1,), generator=rng).item())
                                else:
                                    y0 = int(torch.randint(low=0, high=high_y, size=(1,)).item())
                                    x0 = int(torch.randint(low=0, high=high_x, size=(1,)).item())
                                mask_bw = torch.zeros((H, W), dtype=torch.float32)
                                mask_bw[y0:y0 + side, x0:x0 + side] = 1.0
                            else:
                                raise ValueError(f"Unknown masking_mode '{masking_mode}'. Use 'entropy' or 'random'.")
                        xm = xb[bi].detach().cpu() * (1.0 - mask_bw.detach().cpu().unsqueeze(0))
                        imgs.append(xm)
                        labels.append(f"{lv}%")
                        x_masked_b = xb[bi:bi+1] * (1.0 - mask_bw.to(xb.device).unsqueeze(0).unsqueeze(0))
                        outputs_m, attn_m = model(x_masked_b, return_attn=True)
                        attn_tokens = attn_m.mean(dim=1)[:, 0, 1:]
                        T = attn_tokens.shape[-1]
                        token_side = int(round(math.sqrt(T)))
                        if token_side * token_side != T:
                            token_side = int(math.floor(math.sqrt(T)))
                            attn_tokens = attn_tokens[:, :token_side * token_side]
                        attn_grid = attn_tokens.view(1, 1, token_side, token_side)
                        attn_up = F.interpolate(attn_grid, size=(H, W), mode='bilinear', align_corners=False)[0, 0]
                        attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
                        attn_maps.append(attn_norm.detach().cpu())
                    K = len(imgs)
                    fig, axes = plt.subplots(2, K, figsize=(4*K, 8))
                    for j in range(K):
                        axes[0, j].imshow(imgs[j].permute(1, 2, 0).numpy())
                        base_fs = plt.rcParams.get('font.size', 10.0)
                        axes[0, j].set_title(labels[j], fontsize=base_fs * 2.5, fontweight='bold')
                        axes[0, j].axis('off')
                    for j in range(K):
                        axes[1, j].imshow(attn_maps[j].numpy(), cmap='inferno')
                        axes[1, j].axis('off')
                    tag = example_tag or "examples"
                    out_ex = os.path.join(mask_figs_dir, f"{tag}_idx{global_idx:05d}_{class_name}.png")
                    fig.tight_layout()
                    fig.savefig(out_ex, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    saved_examples += 1
            total += (end - start)
    errors = [1.0 - (correct_per_ratio[r] / total) for r in ratios]
    entropies = [entropy_sum_per_ratio[r] / total for r in ratios]
    spatial_entropies = [spatial_entropy_sum_per_ratio[r] / total for r in ratios]
    freq_energies = [freq_energy_sum_per_ratio[r] / total for r in ratios]
    return ratios, errors, entropies, spatial_entropies, freq_energies


def plot_frequency_energy_trend(ratios, freq_energies, logits_entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    base_fs = plt.rcParams.get('font.size', 10.0)
    fs = base_fs * 2
    line_sp, = ax1.plot(ratios, freq_energies, marker='o', color='tab:purple', label='Frequency Energy (masked region)', linewidth=4)
    ax1.set_xlabel('Masking (%)', fontsize=fs, fontweight='bold')
    ax1.set_ylabel('Frequency Energy', color='tab:purple', fontsize=fs, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:purple', labelsize=fs * 0.9)
    ax1.tick_params(axis='x', labelsize=fs * 0.9)
    ax1.set_xticks(ratios)
    ax2 = ax1.twinx()
    line_log, = ax2.plot(ratios, logits_entropies, marker='s', color='tab:blue', label='Prediction Entropy', linewidth=4)
    ax2.set_ylabel('Prediction Entropy', color='tab:blue', fontsize=fs, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fs * 0.9)
    plt.title(title, fontsize=fs * 1.1, fontweight='bold')
    lines = [line_sp, line_log]
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='best', fontsize=fs)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontweight('bold')
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_spatial_entropy_trend(ratios, spatial_entropies, logits_entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 7))
    base_fs = plt.rcParams.get('font.size', 10.0)
    fs = base_fs * 2
    eps = 1e-12
    base_sp = spatial_entropies[0] if len(spatial_entropies) > 0 else 1.0
    spatial_norm_1m = [1.0 - (s / max(base_sp, eps)) for s in spatial_entropies]
    line_sp, = ax1.plot(ratios, spatial_norm_1m, marker='o', color='tab:green', label='1 - Normalized Spatial Entropy', linewidth=4)
    ax1.set_xlabel('Masking (%)', fontsize=fs, fontweight='bold')
    ax1.set_ylabel('1 - Normalized Spatial Entropy', color='tab:green', fontsize=fs, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:green', labelsize=fs * 0.9)
    ax1.tick_params(axis='x', labelsize=fs * 0.9)
    ax1.set_xticks(ratios)
    ax2 = ax1.twinx()
    line_log, = ax2.plot(ratios, logits_entropies, marker='s', color='tab:blue', label='Prediction Entropy', linewidth=4)
    ax2.set_ylabel('Prediction Entropy', color='tab:blue', fontsize=fs, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fs * 0.9)
    plt.title(title, fontsize=fs * 1.1, fontweight='bold')
    lines = [line_sp, line_log]
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='best', fontsize=fs)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontweight('bold')
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def evaluate_frequency_masking_trend(model: torch.nn.Module,
                                     x: torch.Tensor,
                                     y: torch.Tensor,
                                     device: torch.device,
                                     batch_size: int = 50,
                                     ratios: List[int] = None,
                                     save_mask_examples: int = 0,
                                     mask_example_levels: List[int] = None,
                                     mask_figs_dir: str = None,
                                     example_tag: str = "",
                                     freq_masking_type: Optional[str] = None,
                                     target_class_idx: Optional[int] = None,
                                     model_input_size: Tuple[int, int] = (384, 384),
                                     save_frequency_energy_plot: bool = False) -> Tuple[List[int], List[float], List[float]]:
    """
    Compute error and mean entropy across progressive frequency masking ratios.

    Masking progressively zeros the lowest-energy frequency bins per image,
    always preserving the DC component (0,0). The ratio indicates the percentage
    of non-DC frequency bins that are zeroed.

    Returns: (ratios, errors, entropies)
    """
    model.eval()

    if ratios is None:
        ratios = [i for i in range(0, 101, 10)]

    # Accumulators
    total = 0
    correct_per_ratio = {r: 0 for r in ratios}
    entropy_sum_per_ratio = {r: 0.0 for r in ratios}

    # Defaults for saving examples
    if mask_example_levels is None:
        mask_example_levels = [0, 10, 20]
    saved_examples = 0

    with torch.no_grad():
        N = x.shape[0]
        B_full, C_full, H_full, W_full = x.shape
        tgtH, tgtW = model_input_size

        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            for r in ratios:
                m = float(r)
                # Apply lowest-energy frequency masking and evaluate
                xb_masked = apply_frequency_mask(xb, mask_percent=m, mode=(freq_masking_type or 'lowest_energy'))
                # Resize to model input size right before the forward pass
                if (xb_masked.shape[-2], xb_masked.shape[-1]) != (tgtH, tgtW):
                    xb_in = F.interpolate(xb_masked, size=(tgtH, tgtW), mode='bilinear', align_corners=False)
                else:
                    xb_in = xb_masked
                logits = model(xb_in, return_attn=False)

                pred = logits.argmax(dim=1)
                correct = (pred == yb).sum().item()
                ent = entropy_from_logits(logits).sum().item()

                correct_per_ratio[r] += correct
                entropy_sum_per_ratio[r] += ent

            # Save example figures
            if save_mask_examples > 0 and saved_examples < save_mask_examples and mask_figs_dir is not None:
                os.makedirs(mask_figs_dir, exist_ok=True)
                B_cur = xb.shape[0]
                for bi in range(B_cur):
                    if saved_examples >= save_mask_examples:
                        break
                    # Filter by target class if specified
                    if target_class_idx is not None and int(yb[bi].detach().cpu().item()) != target_class_idx:
                        continue
                    # Only save specified class if requested
                    class_idx = int(yb[bi].detach().cpu().item())
                    if example_tag == "":
                        tag = "examples"
                    else:
                        tag = example_tag

                    imgs = []
                    labels = []
                    attn_maps = []
                    specs = []

                    base_img_cpu = None
                    for lv in mask_example_levels:
                        xm_masked = apply_frequency_mask(xb[bi:bi+1], mask_percent=float(lv), mode=(freq_masking_type or 'lowest_energy'))
                        # For visualization: nearest neighbor to avoid extra blur from interpolation
                        if (xm_masked.shape[-2], xm_masked.shape[-1]) != (tgtH, tgtW):
                            xm_plot = F.interpolate(xm_masked, size=(tgtH, tgtW), mode='nearest')
                        else:
                            xm_plot = xm_masked
                        xm_cpu = xm_plot[0].detach().cpu()
                        imgs.append(xm_cpu)
                        if base_img_cpu is None:
                            base_img_cpu = xm_cpu
                            labels.append(f"{lv}% [{freq_masking_type or 'lowest_energy'}]")
                        else:
                            delta = (xm_cpu - base_img_cpu).abs().mean().item()
                            labels.append(f"{lv}% (Δ={delta:.3f}) [{freq_masking_type or 'lowest_energy'}]")

                        # Attention heatmap
                        # For model: bilinear upsampling
                        if (xm_masked.shape[-2], xm_masked.shape[-1]) != (tgtH, tgtW):
                            xm_in = F.interpolate(xm_masked, size=(tgtH, tgtW), mode='bilinear', align_corners=False)
                        else:
                            xm_in = xm_masked
                        outputs_m, attn_m = model(xm_in.to(device), return_attn=True)
                        attn_tokens = attn_m.mean(dim=1)[:, 0, 1:]  # [1, T]
                        T = attn_tokens.shape[-1]
                        token_side = int(round(math.sqrt(T)))
                        if token_side * token_side != T:
                            token_side = int(math.floor(math.sqrt(T)))
                            attn_tokens = attn_tokens[:, :token_side * token_side]
                        attn_grid = attn_tokens.view(1, 1, token_side, token_side)
                        attn_up = F.interpolate(attn_grid, size=(tgtH, tgtW), mode='bilinear', align_corners=False)[0, 0]
                        attn_norm = (attn_up - attn_up.min()) / (attn_up.max() - attn_up.min() + 1e-8)
                        attn_maps.append(attn_norm.detach().cpu())

                        # Optional: frequency magnitude spectrum visualization (log scale)
                        if save_frequency_energy_plot:
                            Xcur = torch.fft.fft2(xm_masked.to(torch.float32), dim=(-2, -1), norm='ortho')
                            Xcur_c = _fftshift2(Xcur)
                            # Sum over channels, take magnitude
                            mag = (Xcur_c.abs() ** 2).sum(dim=1, keepdim=True)
                            mag = mag[0, 0]  # [H,W]
                            mag = torch.log1p(mag)
                            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                            if (mag.shape[-2], mag.shape[-1]) != (tgtH, tgtW):
                                mag = F.interpolate(mag.unsqueeze(0).unsqueeze(0), size=(tgtH, tgtW), mode='nearest')[0, 0]
                            specs.append(mag.detach().cpu())

                    K = len(imgs)
                    rows = 2
                    fig, axes = plt.subplots(rows, K, figsize=(4*K, 4*rows))
                    base_fs = plt.rcParams.get('font.size', 10.0)
                    for j in range(K):
                        axes[0, j].imshow(imgs[j].permute(1, 2, 0).numpy())
                        # Put the masking level label aligned with each column
                        if j < len(mask_example_levels):
                            axes[0, j].set_title(f"{mask_example_levels[j]}%", fontsize=base_fs * 2.2, fontweight='bold')
                        axes[0, j].axis('off')
                    for j in range(K):
                        axes[1, j].imshow(attn_maps[j].numpy(), cmap='inferno')
                        axes[1, j].axis('off')

                    global_idx = start + bi
                    class_name = CIFAR10_CLASSES[class_idx] if 0 <= class_idx < len(CIFAR10_CLASSES) else f"class{class_idx}"
                    out_ex = os.path.join(mask_figs_dir, f"{tag}_idx{global_idx:05d}_{class_name}.png")
                    fig.tight_layout()
                    fig.savefig(out_ex, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    saved_examples += 1

            total += (end - start)

    errors = [1.0 - (correct_per_ratio[r] / total) for r in ratios]
    entropies = [entropy_sum_per_ratio[r] / total for r in ratios]
    return ratios, errors, entropies


def plot_trend(ratios, errors, entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    base_fs = plt.rcParams.get('font.size', 10.0)
    fs = base_fs * 2

    errors_pct = [e * 100.0 for e in errors]
    line_err, = ax1.plot(ratios, errors_pct, marker='o', color='tab:red', label='Error', linewidth=4)
    ax1.set_xlabel('Masking (%)', fontsize=fs, fontweight='bold')
    ax1.set_ylabel('Error (%)', color='tab:red', fontsize=fs, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=fs * 0.9)
    ax1.tick_params(axis='x', labelsize=fs * 0.9)
    ax1.set_xticks(ratios)
    ax1.set_ylim(0.0, 100.0)

    ax2 = ax1.twinx()
    e0 = errors_pct[0]
    h0 = entropies[0] if len(entropies) > 0 else 1.0
    scale = e0 / h0 if h0 > 1e-12 else 1.0
    ent_scaled = [h * scale for h in entropies]

    line_ent, = ax2.plot(ratios, ent_scaled, marker='s', color='tab:blue', label='Entropy', linewidth=4)
    ax2.set_ylabel('Entropy', color='tab:blue', fontsize=fs, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=fs * 0.9)

    def inv_format(y, pos):
        return f"{(y / max(scale, 1e-12)):.3f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(inv_format))

    if len(ent_scaled) > 0:
        ymin = min(ent_scaled)
        ymax = max(ent_scaled)
        pad = 0.05 * (ymax - ymin + 1e-6)
        ax2.set_ylim(ymin - pad, ymax + pad)

    plt.title(title, fontsize=fs * 1.1, fontweight='bold')
    lines = [line_err, line_ent]
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc='best', fontsize=fs)
    for txt in leg.get_texts():
        txt.set_fontweight('bold')
    for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
        tick.set_fontweight('bold')
    for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
        tick.set_fontweight('bold')
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Masking trend on CIFAR-10-C for source model (no adaptation). Supports spectral (FFT) and spatial masking.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10-C data directory')
    parser.add_argument('--ckpt_dir', type=str, default='/users/doloriel/work/Repo/SPARC/ckpt',
                        help='Checkpoint directory (used by robustbench.load_model)')
    parser.add_argument('--checkpoint', type=str, default='/users/doloriel/work/Repo/SPARC/ckpt/vit_base_384_cifar10.t7',
                        help='Path to ViT-Base 384 checkpoint for CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=10000,
                        help='Number of examples to evaluate per corruption (use 10000 for full)')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=os.path.join(CIFAR_DIR, 'plots'))
    parser.add_argument('--progression', type=int, nargs=3, metavar=('START','STOP','STEP'), default=[0, 100, 10],
                        help='Progression of percentage strengths, e.g., 0 100 5 for 0%,5%,...,100%')
    parser.add_argument('--save_mask_examples', type=int, default=0,
                        help='Number of masked example samples to save per corruption (0 disables)')
    parser.add_argument('--mask_example_levels', type=int, nargs='+', default=[0, 10, 20],
                        help='Masking levels (%) to visualize, e.g., 0 10 20')
    parser.add_argument('--mask_figs_dir', type=str, default=None,
                        help='Directory to save masked example figures (required if saving examples)')
    parser.add_argument('--example_class', type=str, default=None,
                        help='If set, only save example figures for this class (name e.g., "cat" or index 0-9)')
    parser.add_argument('--freq_masking_type', type=str, default='radial_highfreq',
                        choices=['lowest_energy', 'radial_highfreq', 'radial_lowfreq'],
                        help='(Spectral mode only) Frequency masking strategy. Ignored for random spectral masking currently.')
    parser.add_argument('--save_frequency_energy_plot', action='store_true',
                        help='(Spectral mode) If set, add a third row to the saved example figures showing the log-magnitude frequency spectrum after masking (not plotted yet).')
    parser.add_argument('--mask_type', type=str, default='spectral', choices=['spectral', 'spatial'],
                        help='Select masking domain: spectral (FFT-based) or spatial (square-mask based).')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='(Spatial mode) Patch size in pixels for entropy estimation grid; must divide 384 (e.g., 16 => 24x24 grid)')
    parser.add_argument('--entropy_bins', type=int, default=16,
                        help='(Spatial mode) Number of histogram bins for entropy computation')
    parser.add_argument('--use_color_entropy', action='store_true',
                        help='(Spatial mode) Compute entropy on RGB channels (averaged) instead of grayscale')
    parser.add_argument('--entropy_weight_power', type=float, default=2.0,
                        help='(Spatial mode) Power applied to top-entropy weights when computing centroid; >1 emphasizes higher entropies')
    parser.add_argument('--masking_mode', type=str, default='entropy', choices=['entropy', 'random'],
                        help="(Spatial mode) Masking strategy to use: 'entropy' (default) or 'random' for random square placement")
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Optional random seed for reproducible masking (applies to both modes)')
    parser.add_argument('--save_spatial_entropy_plot', action='store_true',
                        help='(Spatial mode) Save plot of spatial image entropy vs prediction logits entropy across masking progression')
    parser.add_argument('--spatial_entropy_bins', type=int, default=64,
                        help='(Spatial mode) Number of histogram bins for spatial image entropy computation')
    args = parser.parse_args()

    # Determine target class index if user specified one
    target_class_idx = None
    if args.example_class is not None:
        sel = str(args.example_class).strip()
        idx = None
        if sel.isdigit():
            idx = int(sel)
        else:
            sel_norm = sel.lower().replace(' ', '').replace('_', '')
            names_norm = [n.lower().replace(' ', '').replace('_', '') for n in CIFAR10_CLASSES]
            if sel_norm in names_norm:
                idx = names_norm.index(sel_norm)
        if idx is None or not (0 <= idx < len(CIFAR10_CLASSES)):
            raise ValueError(f"Invalid example_class '{args.example_class}'. Use 0-9 or one of {CIFAR10_CLASSES}.")
        target_class_idx = idx

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Global seeding for reproducibility across modes
    if args.random_seed is not None:
        torch.manual_seed(int(args.random_seed))

    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    corruption_types = ['gaussian_noise', 'defocus_blur', 'snow', 'jpeg_compression']
    title_map = {
        'gaussian_noise': 'Noise (Gaussian)',
        'defocus_blur': 'Blur (Defocus)',
        'snow': 'Weather (Snow)',
        'jpeg_compression': 'Digital (Jpeg)',
    }

    for ctype in tqdm(corruption_types, desc="Corruptions"):
        x_test, y_test = load_cifar10c(args.num_examples, args.severity, args.data_dir, False, [ctype])

        start, stop, step = args.progression
        ratios_list = list(range(start, stop + (0 if (stop - start) % max(step,1) != 0 else 0) + 1, step))
        if ratios_list[-1] != stop:
            ratios_list.append(stop)

        if args.mask_type == 'spectral':
            # Spectral masking at native 32x32; model resizing done inside evaluate
            ratios, errors, entropies = evaluate_frequency_masking_trend(
                model, x_test, y_test, device,
                batch_size=args.batch_size,
                ratios=ratios_list,
                save_mask_examples=args.save_mask_examples,
                mask_example_levels=args.mask_example_levels,
                mask_figs_dir=args.mask_figs_dir,
                example_tag=ctype,
                freq_masking_type=args.freq_masking_type,
                target_class_idx=target_class_idx,
                save_frequency_energy_plot=args.save_frequency_energy_plot,
            )
            title = title_map.get(ctype, ctype)
            out_name = f'{ctype}_freq_masking_trend.png'
            out_path = os.path.join(args.out_dir, out_name)
            plot_trend(ratios, errors, entropies, title, out_path)
            print(f'Saved plot: {out_path}')
        else:
            # Spatial masking: resize to 384x384 for ViT-B/16-384
            x_in = F.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)
            H = W = 384
            if H % args.patch_size != 0 or W % args.patch_size != 0:
                raise ValueError(f"patch_size {args.patch_size} must evenly divide 384")
            patches_per_side = H // args.patch_size
            rng = None
            if args.random_seed is not None:
                rng = torch.Generator(device='cpu')
                rng.manual_seed(int(args.random_seed))
            ratios, errors, entropies, spatial_entropies, freq_energies = evaluate_entropy_masking_trend(
                model, x_in, y_test, device,
                batch_size=args.batch_size,
                ratios=ratios_list,
                save_mask_examples=args.save_mask_examples,
                mask_example_levels=args.mask_example_levels,
                mask_figs_dir=args.mask_figs_dir,
                example_tag=ctype,
                patches_per_side=patches_per_side,
                num_bins=args.entropy_bins,
                spatial_entropy_bins=args.spatial_entropy_bins,
                use_color_entropy=args.use_color_entropy,
                entropy_weight_power=args.entropy_weight_power,
                target_class_idx=target_class_idx,
                masking_mode=args.masking_mode,
                rng=rng,
            )
            title = title_map.get(ctype, ctype)
            out_name = f'{ctype}_entropy_masking_trend.png'
            out_path = os.path.join(args.out_dir, out_name)
            plot_trend(ratios, errors, entropies, title, out_path)
            print(f'Saved plot: {out_path}')
            if args.save_spatial_entropy_plot:
                out_name_se = f'{ctype}_spatial_entropy_trend.png'
                out_path_se = os.path.join(args.out_dir, out_name_se)
                plot_spatial_entropy_trend(ratios, spatial_entropies, entropies, title + ' – Spatial vs Logits Entropy', out_path_se)
                print(f'Saved plot: {out_path_se}')
            if args.save_frequency_energy_plot:
                out_name_fe = f'{ctype}_frequency_energy_trend.png'
                out_path_fe = os.path.join(args.out_dir, out_name_fe)
                plot_frequency_energy_trend(ratios, freq_energies, entropies, title + ' – Frequency Energy vs Logits Entropy', out_path_fe)
                print(f'Saved plot: {out_path_fe}')


if __name__ == '__main__':
    main()
