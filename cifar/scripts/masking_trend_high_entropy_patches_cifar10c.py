#!/usr/bin/env python3
import os
import sys
import math
import argparse
from collections import OrderedDict
from typing import Optional, List

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
    # Predictive entropy:  -(softmax * log_softmax).sum(1)
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def build_source_model(ckpt_dir: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    arch = 'Standard_VITB_REM'
    model = load_model(arch, ckpt_dir, 'cifar10', ThreatModel.corruptions)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = rm_substr_from_state_dict(state, 'module.')
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def compute_patch_entropy(gray: torch.Tensor, patch_size: int = 16, num_bins: int = 32) -> torch.Tensor:
    """
    Compute Shannon entropy (base-2) for each non-overlapping patch of size patch_size x patch_size
    on grayscale images in [0,1].
    gray: [B,1,H,W]
    Returns: entropies [B, K] where K = (H/ps)*(W/ps)
    """
    B, _, H, W = gray.shape
    ps = patch_size
    assert H % ps == 0 and W % ps == 0, 'H and W must be divisible by patch_size'
    # Unfold to patches: [B, P, K] where P = ps*ps, K = number of patches
    patches = F.unfold(gray, kernel_size=ps, stride=ps)  # [B, P, K]
    P = patches.shape[1]
    K = patches.shape[2]
    # Move dims to [B, K, P]
    patches = patches.permute(0, 2, 1)
    # Clamp to [0,1]
    patches = patches.clamp(0.0, 1.0)
    # Bin indices in [0, num_bins-1]
    idx = torch.clamp((patches * num_bins).long(), 0, num_bins - 1)  # [B,K,P]
    # Count occurrences per bin with scatter_add
    counts = torch.zeros((B, K, num_bins), device=patches.device, dtype=torch.float)
    ones = torch.ones_like(idx, dtype=torch.float)
    counts.scatter_add_(dim=2, index=idx, src=ones)
    # Normalize to probabilities
    p = counts / float(P)
    eps = 1e-12
    ent = -(p * torch.log2(p + eps)).sum(dim=2)  # [B,K]
    return ent


def mask_top_entropy_patches(x: torch.Tensor, entropies: torch.Tensor, ratio: int, patch_size: int = 16) -> torch.Tensor:
    """
    Mask the top-entropy patches at the given ratio (percentage of patches) by zeroing them.
    x: [B,C,H,W], entropies: [B,K], returns masked images [B,C,H,W].
    """
    if ratio <= 0:
        return x
    B, C, H, W = x.shape
    ps = patch_size
    K = (H // ps) * (W // ps)
    m = ratio / 100.0
    k = int(K * m)
    if k <= 0:
        return x

    # Unfold color image: [B, C*ps*ps, K]
    patches = F.unfold(x, kernel_size=ps, stride=ps)
    # For each sample, zero columns corresponding to top-k entropy patches
    topk_idx = torch.topk(entropies, k, dim=1, largest=True).indices  # [B,k]
    for b in range(B):
        cols = topk_idx[b]
        patches[b, :, cols] = 0.0
    # Fold back
    x_masked = F.fold(patches, output_size=(H, W), kernel_size=ps, stride=ps)
    # Values are already filled per patch, no overlap so fold is exact
    x_masked = x_masked.clamp(0.0, 1.0)
    return x_masked


def evaluate_entropy_patch_masking(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 50,
    patch_size: int = 16,
    num_bins: int = 32,
    ratios: Optional[List[int]] = None,
):
    """
    For a corruption tensor (x,y), compute error and mean predictive entropy for masking
    the top-entropy patches progressively by ratios in 0..100.
    Returns: ratios (list[int]), errors (list[float]), entropies (list[float])
    """
    model.eval()
    if ratios is None:
        ratios = [i for i in range(0, 101, 5)]

    total = 0
    correct_per_ratio = {r: 0 for r in ratios}
    entropy_sum_per_ratio = {r: 0.0 for r in ratios}

    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc='Batches', leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            # Compute per-patch entropies on grayscale
            gray = 0.2989 * xb[:, 0:1] + 0.5870 * xb[:, 1:2] + 0.1140 * xb[:, 2:3]
            ent = compute_patch_entropy(gray, patch_size=patch_size, num_bins=num_bins)  # [B,K]

            for r in ratios:
                if r == 0:
                    x_proc = xb
                else:
                    x_proc = mask_top_entropy_patches(xb, ent, ratio=r, patch_size=patch_size)

                logits = model(x_proc, return_attn=False)
                pred = logits.argmax(dim=1)
                correct = (pred == yb).sum().item()
                ent_pred = entropy_from_logits(logits).sum().item()

                correct_per_ratio[r] += correct
                entropy_sum_per_ratio[r] += ent_pred

            total += (end - start)

    errors = [1.0 - (correct_per_ratio[r] / total) for r in ratios]
    entropies = [entropy_sum_per_ratio[r] / total for r in ratios]
    return ratios, errors, entropies


def plot_trend(ratios, errors, entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left axis: error (%)
    errors_pct = [e * 100.0 for e in errors]
    ax1.plot(ratios, errors_pct, marker='o', color='tab:red', label='Error')
    ax1.set_xlabel('Top high-entropy patches masked (%)')
    ax1.set_ylabel('Error (%)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(ratios)
    ax1.set_ylim(0.0, 100.0)

    # Right axis: entropy (scaled for visual alignment at 0%)
    ax2 = ax1.twinx()
    e0 = errors_pct[0]
    h0 = entropies[0] if len(entropies) > 0 else 1.0
    scale = e0 / h0 if h0 > 1e-12 else 1.0
    ent_scaled = [h * scale for h in entropies]

    ax2.plot(ratios, ent_scaled, marker='s', color='tab:blue', label='Entropy')
    ax2.set_ylabel('Entropy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    def inv_format(y, pos):
        return f"{(y / max(scale, 1e-12)):.3f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(inv_format))

    if len(ent_scaled) > 0:
        ymin = min(ent_scaled)
        ymax = max(ent_scaled)
        pad = 0.05 * (ymax - ymin + 1e-6)
        ax2.set_ylim(ymin - pad, ymax + pad)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def save_entropy_mask_examples(
    x1: torch.Tensor,
    device: torch.device,
    patch_size: int,
    num_bins: int,
    levels: list,
    figs_dir: str,
    example_tag: str,
):
    """
    Save a 1xK grid of masked images for given percentage levels using the
    entropy-based top-patch masking. x1 is a single sample [1,C,H,W].
    """
    os.makedirs(figs_dir, exist_ok=True)

    with torch.no_grad():
        x1 = x1.to(device)
        # Compute entropies for the single sample
        gray1 = 0.2989 * x1[:, 0:1] + 0.5870 * x1[:, 1:2] + 0.1140 * x1[:, 2:3]
        ent1 = compute_patch_entropy(gray1, patch_size=patch_size, num_bins=num_bins)  # [1,K]

        imgs = []
        labels = []
        for lv in levels:
            if lv <= 0:
                xr = x1
            else:
                xr = mask_top_entropy_patches(x1, ent1, ratio=int(lv), patch_size=patch_size)
            imgs.append(xr[0].detach().cpu())
            labels.append(f"{lv}%")

        K = len(imgs)
        fig, axes = plt.subplots(1, K, figsize=(3*K, 3))
        if K == 1:
            axes = [axes]
        for ax, im, lab in zip(axes, imgs, labels):
            ax.imshow(im.permute(1, 2, 0).numpy())
            ax.set_title(lab)
            ax.axis('off')
        out_ex = os.path.join(figs_dir, f"{example_tag}_entropy_mask_examples.png")
        fig.tight_layout()
        fig.savefig(out_ex, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Mask top high-entropy patches (16x16) progressively on CIFAR-10-C and plot error/entropy.'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10-C data directory')
    parser.add_argument('--ckpt_dir', type=str, default=os.path.join(CIFAR_DIR, '..', 'ckpt'),
                        help='Checkpoint directory (used by robustbench.load_model)')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(CIFAR_DIR, '..', 'ckpt', 'vit_base_384_cifar10.t7'),
                        help='Path to ViT-Base 384 checkpoint for CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=10000,
                        help='Number of examples to evaluate per corruption (use 10000 for full)')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=os.path.join(CIFAR_DIR, 'plots'))
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_bins', type=int, default=32)
    parser.add_argument('--progression', type=int, nargs=3, metavar=('START','STOP','STEP'), default=[0, 100, 5],
                        help='Masking ratios in percent, e.g., 0 100 5 for 0%,5%,...,100%')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Which CIFAR-10-C corruptions to evaluate (default uses title_map keys)')
    parser.add_argument('--save_examples', type=int, default=1,
                        help='Number of example samples to save per corruption (0 disables)')
    parser.add_argument('--example_levels', type=int, nargs='+', default=[0, 10, 20],
                        help='Masking levels (%) to visualize, e.g., 0 10 20')
    parser.add_argument('--figs_dir', type=str, default='/users/doloriel/work/Repo/FreqREM/cifar/figs/EntREM',
                        help='Directory to save masked view example figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    # Title map and default corruptions
    title_map = {
        'gaussian_noise': 'Noise (Gaussian)',
        'defocus_blur': 'Blur (Defocus)',
        'snow': 'Weather (Snow)',
        'jpeg_compression': 'Digital (Jpeg)',
    }
    if args.corruptions is None:
        args.corruptions = list(title_map.keys())

    for ctype in tqdm(args.corruptions, desc='Corruptions'):
        x_test, y_test = load_cifar10c(args.num_examples, args.severity, args.data_dir, False, [ctype])
        # Resize to 384x384 for ViT-B/16-384
        x_test = F.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)

        # Save masked view examples (one sample per corruption) at requested levels
        if args.save_examples > 0 and x_test.shape[0] > 0:
            to_take = min(args.save_examples, x_test.shape[0])
            for i in range(to_take):
                x1 = x_test[i:i+1]
                save_entropy_mask_examples(
                    x1, device,
                    patch_size=args.patch_size,
                    num_bins=args.num_bins,
                    levels=args.example_levels,
                    figs_dir=args.figs_dir,
                    example_tag=ctype,
                )

        start, stop, step = args.progression
        ratios_list = list(range(start, stop + (0 if (stop - start) % max(step,1) != 0 else 0) + 1, step))
        if ratios_list[-1] != stop:
            ratios_list.append(stop)

        ratios, errors, entropies = evaluate_entropy_patch_masking(
            model, x_test, y_test, device,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_bins=args.num_bins,
            ratios=ratios_list,
        )

        title = title_map.get(ctype, ctype)
        out_name = f'{ctype}_high_entropy_patches_masking.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(ratios, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
