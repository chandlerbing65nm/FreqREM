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


def evaluate_masking_trend(model: torch.nn.Module,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           device: torch.device,
                           batch_size: int = 50,
                           mode: str = 'mask',
                           phase_seed: int = 0,
                           ratios: list = None,
                           save_phase_examples: int = 0,
                           phase_example_levels: list = None,
                           figs_dir: str = None,
                           example_tag: str = ""):
    """
    For a single corruption type tensor (x,y): compute error and mean entropy
    for masking ratios from 0% to 100% in steps of 10% using the same procedure
    as in cifar/rem.py.
    Returns: ratios (list of ints 0..100), errors (list), entropies (list)
    """
    model.eval()

    # ViT-B/16 @ 384x384 has 24*24 = 576 patch tokens (used in mask mode)
    tokens = 576
    # Strength levels (percent). Default: 0,10,...,100 if not provided
    if ratios is None:
        ratios = [i for i in range(0, 101, 10)]

    # Accumulators
    total = 0
    correct_per_ratio = {r: 0 for r in ratios}
    entropy_sum_per_ratio = {r: 0.0 for r in ratios}

    # Defaults for example saving
    if phase_example_levels is None:
        phase_example_levels = [0, 5, 10]
    saved_examples = 0

    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            if mode == 'mask':
                # Unmasked forward to get reference logits and attention
                outputs, attn = model(xb, return_attn=True)  # logits: [B,C], attn: [B, H, 577, 577]
                attn_score = attn.mean(dim=1)[:, 0, 1:]      # [B, 576]

                # For each masking ratio, compute kept indices and forward
                for r in ratios:
                    if r == 0:
                        logits = outputs  # reuse unmasked
                    else:
                        m = r / 100.0
                        num_keep = int(tokens * (1.0 - m))
                        if num_keep > 0:
                            # Select smallest attention values to keep, mirroring cifar/rem.py
                            len_keep = torch.topk(attn_score, num_keep, largest=False).indices
                        else:
                            # Keep zero patch tokens; rem_vit will keep only CLS token
                            len_keep = attn_score[:, :0]
                        logits = model(xb, len_keep=len_keep, return_attn=False)
                    
                    # Accuracy & entropy
                    pred = logits.argmax(dim=1)
                    correct = (pred == yb).sum().item()
                    ent = entropy_from_logits(logits).sum().item()

                    correct_per_ratio[r] += correct
                    entropy_sum_per_ratio[r] += ent
            else:
                # Phase distortion mode: progressively randomize FFT phase (norm='ortho')
                # Create a fixed random phase per-batch so higher strengths move further towards the same target
                torch.manual_seed(phase_seed)
                B, C, H, W = xb.shape
                psi = (torch.rand((B, C, H, W), device=xb.device) * 2 * math.pi) - math.pi  # [-pi, pi]
                unit_rand = torch.polar(torch.ones_like(psi), psi)  # e^{i psi}

                for r in ratios:
                    alpha = r / 100.0  # 0..1 strength
                    # FFT
                    X = torch.fft.fft2(xb, dim=(-2, -1), norm='ortho')
                    mag = torch.abs(X)
                    eps = 1e-8
                    unit = X / (mag + eps)  # e^{i phi}
                    # Mix original and random phase, then renormalize to unit magnitude
                    mixed = (1.0 - alpha) * unit + alpha * unit_rand
                    mixed = mixed / (mixed.abs() + eps)
                    Xp = mag * mixed
                    x_rec = torch.fft.ifft2(Xp, dim=(-2, -1), norm='ortho').real
                    x_rec = x_rec.clamp(0.0, 1.0)

                    logits = model(x_rec, return_attn=False)

                    pred = logits.argmax(dim=1)
                    correct = (pred == yb).sum().item()
                    ent = entropy_from_logits(logits).sum().item()

                    correct_per_ratio[r] += correct
                    entropy_sum_per_ratio[r] += ent

                # Optionally save example figures for a few samples at fixed levels (e.g., 0,5,10%)
                if save_phase_examples > 0 and saved_examples < save_phase_examples and figs_dir is not None:
                    import matplotlib.pyplot as plt
                    os.makedirs(figs_dir, exist_ok=True)

                    # How many from this batch to save
                    to_take = min(save_phase_examples - saved_examples, B)
                    for bi in range(to_take):
                        # For each requested level, build distorted sample
                        imgs = []
                        labels = []
                        x1 = xb[bi:bi+1]
                        ur1 = unit_rand[bi:bi+1]
                        # Precompute FFT, mag and unit phase for efficiency
                        X = torch.fft.fft2(x1, dim=(-2, -1), norm='ortho')
                        mag = torch.abs(X)
                        eps = 1e-8
                        unit = X / (mag + eps)
                        for lv in phase_example_levels:
                            alpha = float(lv) / 100.0
                            mixed = (1.0 - alpha) * unit + alpha * ur1
                            mixed = mixed / (mixed.abs() + eps)
                            Xp = mag * mixed
                            xr = torch.fft.ifft2(Xp, dim=(-2, -1), norm='ortho').real
                            xr = xr.clamp(0.0, 1.0)
                            imgs.append(xr[0].detach().cpu())
                            labels.append(f"{lv}%")

                        # Build a 1xK figure
                        K = len(imgs)
                        fig, axes = plt.subplots(1, K, figsize=(3*K, 3))
                        if K == 1:
                            axes = [axes]
                        for ax, im, lab in zip(axes, imgs, labels):
                            ax.imshow(im.permute(1, 2, 0).numpy())
                            ax.set_title(lab)
                            ax.axis('off')
                        tag = example_tag or "examples"
                        out_ex = os.path.join(figs_dir, f"{tag}_sample{saved_examples+1}.png")
                        fig.tight_layout()
                        fig.savefig(out_ex, dpi=200)
                        plt.close(fig)
                        saved_examples += 1

            total += (end - start)

    errors = [1.0 - (correct_per_ratio[r] / total) for r in ratios]
    entropies = [entropy_sum_per_ratio[r] / total for r in ratios]
    return ratios, errors, entropies


def plot_trend(ratios, errors, entropies, title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Left axis: error (%) to mimic paper
    errors_pct = [e * 100.0 for e in errors]
    ax1.plot(ratios, errors_pct, marker='o', color='tab:red', label='Error')
    ax1.set_xlabel('Distortion/Masking (%)')
    ax1.set_ylabel('Error (%)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(ratios)
    ax1.set_ylim(0.0, 100.0)

    # Right axis: entropy
    ax2 = ax1.twinx()
    # Scale entropy so it visually aligns with error at 0% masking
    e0 = errors_pct[0]
    h0 = entropies[0] if len(entropies) > 0 else 1.0
    scale = e0 / h0 if h0 > 1e-12 else 1.0
    ent_scaled = [h * scale for h in entropies]

    ax2.plot(ratios, ent_scaled, marker='s', color='tab:blue', label='Entropy')
    ax2.set_ylabel('Entropy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Format right-axis ticks to show true entropy values
    def inv_format(y, pos):
        return f"{(y / max(scale, 1e-12)):.3f}"
    ax2.yaxis.set_major_formatter(FuncFormatter(inv_format))

    # Set right-axis limits based on scaled entropy with a small margin
    if len(ent_scaled) > 0:
        ymin = min(ent_scaled)
        ymax = max(ent_scaled)
        pad = 0.05 * (ymax - ymin + 1e-6)
        ax2.set_ylim(ymin - pad, ymax + pad)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Masking trend on CIFAR-10-C for source model (no adaptation).')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10-C data directory')
    parser.add_argument('--ckpt_dir', type=str, default='/users/doloriel/work/Repo/FreqREM/ckpt',
                        help='Checkpoint directory (used by robustbench.load_model)')
    parser.add_argument('--checkpoint', type=str, default='/users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7',
                        help='Path to ViT-Base 384 checkpoint for CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=10000,
                        help='Number of examples to evaluate per corruption (use 10000 for full)')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=os.path.join(CIFAR_DIR, 'plots'))
    parser.add_argument('--mode', type=str, choices=['mask', 'phase'], default='mask',
                        help='mask: ViT token masking (REM-like), phase: progressive FFT phase distortion')
    parser.add_argument('--phase_seed', type=int, default=0,
                        help='Random seed for phase distortion (deterministic across runs)')
    parser.add_argument('--progression', type=int, nargs=3, metavar=('START','STOP','STEP'), default=[0, 100, 10],
                        help='Progression of percentage strengths, e.g., 0 100 5 for 0%,5%,...,100%')
    parser.add_argument('--save_phase_examples', type=int, default=3,
                        help='Number of example samples to save per corruption in phase mode (0 disables)')
    parser.add_argument('--phase_example_levels', type=int, nargs='+', default=[0, 5, 10],
                        help='Phase distortion levels (%) to visualize, e.g., 0 5 10')
    parser.add_argument('--figs_dir', type=str, default=os.path.join(CIFAR_DIR, 'figs'),
                        help='Directory to save example figures')
    args = parser.parse_args()

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
        # Build ratios including the stop value if divisible
        ratios_list = list(range(start, stop + (0 if (stop - start) % max(step,1) != 0 else 0) + 1, step))
        if ratios_list[-1] != stop:
            ratios_list.append(stop)

        ratios, errors, entropies = evaluate_masking_trend(
            model, x_test, y_test, device,
            batch_size=args.batch_size,
            mode=args.mode,
            phase_seed=args.phase_seed,
            ratios=ratios_list,
            save_phase_examples=(args.save_phase_examples if args.mode == 'phase' else 0),
            phase_example_levels=args.phase_example_levels,
            figs_dir=args.figs_dir,
            example_tag=ctype,
        )

        title = title_map.get(ctype, ctype)
        out_name = f'{ctype}_masking_trend.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(ratios, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
