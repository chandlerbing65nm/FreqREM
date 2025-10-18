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
                         mask_percent: float) -> torch.Tensor:
    """
    Apply a frequency-domain mask that progressively zeros the lowest-energy
    frequency bins per image, while always preserving the DC component.

    x: [B,C,H,W] in [0,1]
    mask_percent: percentage (0-100) of non-DC frequency bins to zero.
    Returns masked image [B,C,H,W] in [0,1].
    """
    assert 0.0 <= mask_percent <= 100.0
    B, C, H, W = x.shape

    # Forward FFT (complex). No shift needed; DC is at index (0,0).
    X = torch.fft.fft2(x.to(torch.float32), dim=(-2, -1), norm='ortho')  # [B,C,H,W]

    # Compute power per frequency bin summed over channels: [B,H,W]
    power = (X.abs() ** 2).sum(dim=1)

    # Exclude DC from masking by assigning it +inf power so it is never among lowest energies
    # DC index in unshifted spectrum is (0,0).
    power[:, 0, 0] = float('inf')

    # Number of bins to mask (excluding DC)
    total_bins = H * W - 1
    k = int(round((mask_percent / 100.0) * total_bins))
    if k <= 0:
        X_masked = X
    else:
        # Find indices of k lowest-energy bins per image
        flat_power = power.view(B, -1)  # [B, H*W]
        # Note: DC at flat index 0 is already set to inf and won't be selected
        vals, idxs = torch.topk(flat_power, k, dim=1, largest=False, sorted=False)  # [B, k]

        # Build per-image keep mask in frequency domain: start with ones, set selected indices to 0 (mask)
        keep = torch.ones((B, H * W), device=x.device, dtype=X.dtype)
        keep.scatter_(dim=1, index=idxs, src=torch.zeros_like(idxs, dtype=X.dtype))
        keep = keep.view(B, 1, H, W)  # broadcast to channels

        # Apply mask to all channels
        X_masked = X * keep

    # Inverse FFT back to image space
    x_rec = torch.fft.ifft2(X_masked, dim=(-2, -1), norm='ortho').real
    x_rec = x_rec.clamp(0.0, 1.0)
    return x_rec


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
                                     target_class_idx: Optional[int] = None) -> Tuple[List[int], List[float], List[float]]:
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

        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            for r in ratios:
                m = float(r)
                # Apply lowest-energy frequency masking and evaluate
                xb_masked = apply_frequency_mask(xb, mask_percent=m)
                logits = model(xb_masked, return_attn=False)

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

                    for lv in mask_example_levels:
                        xm = apply_frequency_mask(xb[bi:bi+1], mask_percent=float(lv))
                        xm_cpu = xm[0].detach().cpu()
                        imgs.append(xm_cpu)
                        labels.append(f"{lv}%")

                        # Attention heatmap
                        outputs_m, attn_m = model(xm.to(device), return_attn=True)
                        attn_tokens = attn_m.mean(dim=1)[:, 0, 1:]  # [1, T]
                        T = attn_tokens.shape[-1]
                        token_side = int(round(math.sqrt(T)))
                        if token_side * token_side != T:
                            token_side = int(math.floor(math.sqrt(T)))
                            attn_tokens = attn_tokens[:, :token_side * token_side]
                        attn_grid = attn_tokens.view(1, 1, token_side, token_side)
                        attn_up = F.interpolate(attn_grid, size=(H_full, W_full), mode='bilinear', align_corners=False)[0, 0]
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
    parser = argparse.ArgumentParser(description='Frequency-masked trend on CIFAR-10-C for source model (no adaptation).')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10-C data directory')
    parser.add_argument('--ckpt_dir', type=str, default='/users/doloriel/work/Repo/SPARE/ckpt',
                        help='Checkpoint directory (used by robustbench.load_model)')
    parser.add_argument('--checkpoint', type=str, default='/users/doloriel/work/Repo/SPARE/ckpt/vit_base_384_cifar10.t7',
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

    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    corruption_types = ['gaussian_noise', 'defocus_blur', 'snow', 'jpeg_compression']
    title_map = {
        'gaussian_noise': 'Noise (Gaussian)',
        'defocus_blur': 'Blur (Defocus)',
        'snow': 'Weather (Snow)',
        'jpeg_compression': 'Digital (Jpeg)',
    }

    for ctype in tqdm(corruption_types, desc="Corruptions"):
        # Load the specified corruption at given severity
        x_test, y_test = load_cifar10c(args.num_examples, args.severity, args.data_dir, False, [ctype])
        # Resize to 384x384 for ViT-B/16-384
        x_test = F.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)

        start, stop, step = args.progression
        ratios_list = list(range(start, stop + (0 if (stop - start) % max(step,1) != 0 else 0) + 1, step))
        if ratios_list[-1] != stop:
            ratios_list.append(stop)

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
        )

        title = title_map.get(ctype, ctype)
        out_name = f'{ctype}_freq_masking_trend.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(ratios, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
