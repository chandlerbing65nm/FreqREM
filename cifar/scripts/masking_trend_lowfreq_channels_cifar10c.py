#!/usr/bin/env python3
import os
import sys
import math
import argparse
from collections import OrderedDict

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
    # Same as cifar/rem.py Entropy:  -(softmax * log_softmax).sum(1)
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def build_source_model(ckpt_dir: str, checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    # Use REM-capable ViT so we can request attention and masking via len_keep
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


def build_lowfreq_mask_unshifted_first_half(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Low-frequency mask in the unshifted FFT domain.
    Following the instruction to "divide the unshifted magnitude to 2 and take the first part",
    we mark the first half of the coefficients in row-major order as low-frequency.
    Concretely, this corresponds to the top half of rows: indices i < H//2 for all columns j.

    Returns a boolean tensor of shape (H, W) with True where frequency is considered low.
    """
    mask = torch.zeros((h, w), dtype=torch.bool, device=device)
    mask[: h // 2, :] = True
    return mask


def evaluate_phase_channel_distortion(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 50,
    channel_order = (0, 1, 2),
    phase_seed: int = 0,
    phase_alpha: float = 1.0,
):
    """
    Evaluate error and mean entropy as we progressively distort the FFT phase
    for an increasing number of channels.

    Steps: 0 (original), 1 (distort channel 0), 2 (distort channels 0,1), 3 (distort channels 0,1,2)
    Distortion mixes original unit phase with random unit complex by alpha in [0,1]:
        unit_new = normalize((1-alpha)*unit + alpha*unit_rand)

    Returns: steps (list[int]), errors (list[float]), entropies (list[float]).
    """
    model.eval()

    steps = [0, 1, 2, 3]
    total = 0
    correct_per_step = {s: 0 for s in steps}
    entropy_sum_per_step = {s: 0.0 for s in steps}

    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            B, C, H, W = xb.shape
            assert C >= len(channel_order), "Input should have at least as many channels as channel_order length"

            # Precompute FFT for the batch once
            X = torch.fft.fft2(xb, dim=(-2, -1), norm='ortho')  # complex, shape (B,C,H,W)
            mag = torch.abs(X)
            eps = 1e-8
            unit = X / (mag + eps)  # unit complex with original phase

            # Deterministic random unit complex per batch for reproducibility
            torch.manual_seed(phase_seed)
            psi = (torch.rand((B, C, H, W), device=xb.device) * 2 * math.pi) - math.pi
            unit_rand = torch.polar(torch.ones_like(psi), psi)  # e^{i psi}

            for s in steps:
                if s == 0 or phase_alpha <= 0.0:
                    x_proc = xb
                else:
                    # Mix phases only for the first s channels in channel_order
                    new_unit = unit.clone()
                    for k in range(min(s, len(channel_order))):
                        ch = channel_order[k]
                        mixed = (1.0 - phase_alpha) * unit[:, ch] + phase_alpha * unit_rand[:, ch]
                        new_unit[:, ch] = mixed
                    # Renormalize to unit magnitude to ensure pure phase
                    new_unit = new_unit / (new_unit.abs() + eps)
                    Xp = mag * new_unit
                    x_proc = torch.fft.ifft2(Xp, dim=(-2, -1), norm='ortho').real
                    x_proc = x_proc.clamp(0.0, 1.0)

                logits = model(x_proc, return_attn=False)
                pred = logits.argmax(dim=1)
                correct = (pred == yb).sum().item()
                ent = entropy_from_logits(logits).sum().item()

                correct_per_step[s] += correct
                entropy_sum_per_step[s] += ent

            total += (end - start)

    errors = [1.0 - (correct_per_step[s] / total) for s in steps]
    entropies = [entropy_sum_per_step[s] / total for s in steps]
    return steps, errors, entropies


def plot_trend(steps, errors, entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left axis: error (%)
    errors_pct = [e * 100.0 for e in errors]
    ax1.plot(steps, errors_pct, marker='o', color='tab:red', label='Error')
    ax1.set_xlabel('Number of channels phase-distorted')
    ax1.set_xticks(steps)
    ax1.set_ylabel('Error (%)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_ylim(0.0, 100.0)

    # Right axis: entropy (scaled for visual alignment at step 0)
    ax2 = ax1.twinx()
    e0 = errors_pct[0]
    h0 = entropies[0] if len(entropies) > 0 else 1.0
    scale = e0 / h0 if h0 > 1e-12 else 1.0
    ent_scaled = [h * scale for h in entropies]

    ax2.plot(steps, ent_scaled, marker='s', color='tab:blue', label='Entropy')
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


def save_phase_channel_progression_example(
    x1: torch.Tensor,
    device: torch.device,
    channel_order,
    phase_seed: int,
    phase_alpha: float,
    figs_dir: str,
    example_tag: str,
):
    """
    Save a 1x4 grid of images showing progression for steps 0,1,2,3 channels phase-distorted.
    x1 is a single-sample tensor of shape (1, C, H, W) on CPU or device.
    """
    os.makedirs(figs_dir, exist_ok=True)

    with torch.no_grad():
        x1 = x1.to(device)
        B, C, H, W = x1.shape
        assert B == 1, "x1 should be a single sample with shape (1,C,H,W)"

        # Precompute FFT and unit phases
        X = torch.fft.fft2(x1, dim=(-2, -1), norm='ortho')
        mag = torch.abs(X)
        eps = 1e-8
        unit = X / (mag + eps)

        # Deterministic random unit complex for reproducibility
        torch.manual_seed(phase_seed)
        psi = (torch.rand((B, C, H, W), device=x1.device) * 2 * math.pi) - math.pi
        unit_rand = torch.polar(torch.ones_like(psi), psi)

        imgs = []
        labels = []
        steps = [0, 1, 2, 3]
        for s in steps:
            if s == 0 or phase_alpha <= 0.0:
                xr = x1
            else:
                new_unit = unit.clone()
                for k in range(min(s, len(channel_order))):
                    ch = channel_order[k]
                    mixed = (1.0 - phase_alpha) * unit[:, ch] + phase_alpha * unit_rand[:, ch]
                    new_unit[:, ch] = mixed
                new_unit = new_unit / (new_unit.abs() + eps)
                Xp = mag * new_unit
                xr = torch.fft.ifft2(Xp, dim=(-2, -1), norm='ortho').real
                xr = xr.clamp(0.0, 1.0)

            imgs.append(xr[0].detach().cpu())
            labels.append(f"{s} ch")

        # Build a 1x4 figure
        K = len(imgs)
        fig, axes = plt.subplots(1, K, figsize=(3*K, 3))
        if K == 1:
            axes = [axes]
        for ax, im, lab in zip(axes, imgs, labels):
            ax.imshow(im.permute(1, 2, 0).numpy())
            ax.set_title(lab)
            ax.axis('off')
        out_ex = os.path.join(figs_dir, f"{example_tag}_phase_channels_example.png")
        fig.tight_layout()
        fig.savefig(out_ex, dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Distort FFT phase per-channel progressively on CIFAR-10-C and plot error/entropy.'
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
    parser.add_argument('--channel_order', type=int, nargs='+', default=[0,1,2],
                        help='Order of channels to distort (0-based). Example: 2 1 0')
    parser.add_argument('--phase_seed', type=int, default=0,
                        help='Random seed for phase distortion (deterministic across runs)')
    parser.add_argument('--phase_alpha', type=float, default=1.0,
                        help='Phase mixing strength in [0,1]; 1.0 replaces with random phase, 0.5 mixes evenly')
    parser.add_argument('--corruptions', type=str, nargs='+', default=None,
                        help='Which CIFAR-10-C corruptions to evaluate (default uses title_map keys)')
    parser.add_argument('--save_examples', type=int, default=1,
                        help='Number of example samples to save per corruption (0 disables)')
    parser.add_argument('--figs_dir', type=str, default='/users/doloriel/work/Repo/FreqREM/cifar/figs',
                        help='Directory to save example figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    # Use requested mapping for corruption labels and default corruption set
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

        # Save sample example(s) showing progression across channels
        if args.save_examples > 0 and x_test.shape[0] > 0:
            to_take = min(args.save_examples, x_test.shape[0])
            for i in range(to_take):
                x1 = x_test[i:i+1]
                save_phase_channel_progression_example(
                    x1, device,
                    channel_order=tuple(args.channel_order),
                    phase_seed=args.phase_seed,
                    phase_alpha=args.phase_alpha,
                    figs_dir=args.figs_dir,
                    example_tag=ctype,
                )

        steps, errors, entropies = evaluate_phase_channel_distortion(
            model, x_test, y_test, device,
            batch_size=args.batch_size,
            channel_order=tuple(args.channel_order),
            phase_seed=args.phase_seed,
            phase_alpha=args.phase_alpha,
        )

        title = title_map.get(ctype, ctype)
        out_name = f'{ctype}_phase_channels_progression.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(steps, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
