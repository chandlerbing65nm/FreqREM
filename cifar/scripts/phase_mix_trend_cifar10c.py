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
    # Use REM-capable ViT so we can reuse identical model loading as masking script
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


def phase_attenuate(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Phase attenuation: shrink the Fourier phase angle by a factor (1 - alpha)
    where alpha in [0,1].
      - alpha = 0.0 => identity (no attenuation)
      - alpha = 1.0 => zero phase (pure magnitude image)
    Uses orthonormal FFT to preserve scale.
    """
    # x: [B,C,H,W] in [0,1]
    X = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    mag = torch.abs(X)
    eps = 1e-8
    unit = X / (mag + eps)  # e^{i phi}
    phi = torch.angle(unit)
    phi_att = (1.0 - alpha) * phi
    unit_att = torch.polar(torch.ones_like(phi), phi_att)  # magnitude 1, attenuated phase
    X_att = mag * unit_att
    x_rec = torch.fft.ifft2(X_att, dim=(-2, -1), norm='ortho').real
    x_rec = x_rec.clamp(0.0, 1.0)
    return x_rec


def evaluate_phase_mix_trend(model: torch.nn.Module,
                              x: torch.Tensor,
                              y: torch.Tensor,
                              device: torch.device,
                              batch_size: int = 50,
                              ratios: list = None,
                              phase_alpha: float = 1.0,
                              save_mix_examples: int = 0,
                              mix_example_levels: list = None,
                              figs_dir: str = None,
                              example_tag: str = ""):
    """
    For a single corruption type tensor (x,y):
      1) Build a phase-attenuated counterpart x_phase using attenuation alpha.
      2) For t in ratios (%), mix: x_t = (1 - t) * x + t * x_phase.
      3) Compute error and mean predictive entropy at each t.

    Returns: ratios (list of ints 0..100), errors (list), entropies (list)
    """
    model.eval()

    if ratios is None:
        ratios = [i for i in range(0, 101, 10)]

    if mix_example_levels is None:
        mix_example_levels = [0, 25, 50, 75, 100]

    total = 0
    correct_per_ratio = {r: 0 for r in ratios}
    entropy_sum_per_ratio = {r: 0.0 for r in ratios}

    saved_examples = 0

    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)

            # Compute phase-attenuated counterpart once per batch
            x_phase = phase_attenuate(xb, alpha=phase_alpha)

            for r in ratios:
                t = r / 100.0
                xt = (1.0 - t) * xb + t * x_phase
                xt = xt.clamp(0.0, 1.0)

                logits = model(xt, return_attn=False)
                pred = logits.argmax(dim=1)
                correct = (pred == yb).sum().item()
                ent = entropy_from_logits(logits).sum().item()

                correct_per_ratio[r] += correct
                entropy_sum_per_ratio[r] += ent

            # Optionally save example mixes
            if save_mix_examples > 0 and saved_examples < save_mix_examples and figs_dir is not None:
                os.makedirs(figs_dir, exist_ok=True)
                to_take = min(save_mix_examples - saved_examples, xb.shape[0])
                for bi in range(to_take):
                    imgs = []
                    labels = []
                    x1 = xb[bi:bi+1]
                    x1_phase = phase_attenuate(x1, alpha=phase_alpha)
                    for lv in mix_example_levels:
                        t = float(lv) / 100.0
                        xt = (1.0 - t) * x1 + t * x1_phase
                        xt = xt.clamp(0.0, 1.0)
                        imgs.append(xt[0].detach().cpu())
                        labels.append(f"t={lv}%")

                    K = len(imgs)
                    fig, axes = plt.subplots(1, K, figsize=(3*K, 3))
                    if K == 1:
                        axes = [axes]
                    for ax, im, lab in zip(axes, imgs, labels):
                        ax.imshow(im.permute(1, 2, 0).numpy())
                        ax.set_title(lab)
                        ax.axis('off')
                    tag = example_tag or "phase_mix"
                    out_ex = os.path.join(figs_dir, f"{tag}.png")
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

    # Left axis: error (%)
    errors_pct = [e * 100.0 for e in errors]
    ax1.plot(ratios, errors_pct, marker='o', color='tab:red', label='Error')
    ax1.set_xlabel('Mixing t (%)')
    ax1.set_ylabel('Error (%)', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(ratios)
    ax1.set_ylim(0.0, 100.0)

    # Right axis: entropy (scaled to align at t=0 visually)
    ax2 = ax1.twinx()
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
    parser = argparse.ArgumentParser(description='Phase-mix trend on CIFAR-10-C for source model (no adaptation).')
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
    parser.add_argument('--progression', type=int, nargs=3, metavar=('START','STOP','STEP'), default=[0, 100, 10],
                        help='Progression of mixing t values in percent, e.g., 0 100 5 for 0%,5%,...,100%')
    parser.add_argument('--phase_alpha', type=float, default=1.0,
                        help='Phase attenuation strength alpha in [0,1]; 1.0 = zero phase (magnitude-only) counterpart')
    parser.add_argument('--save_mix_examples', type=int, default=3,
                        help='Number of example samples to save per corruption (0 disables)')
    parser.add_argument('--mix_example_levels', type=int, nargs='+', default=[0, 25, 50, 75, 100],
                        help='Mixing levels t (%) to visualize, e.g., 0 25 50 75 100')
    parser.add_argument('--figs_dir', type=str, default=os.path.join(CIFAR_DIR, 'figs'),
                        help='Directory to save example figures')
    parser.add_argument('--corruptions', type=str, nargs='+', default=['gaussian_noise', 'defocus_blur', 'snow', 'jpeg_compression'],
                        help='List of CIFAR-10-C corruption types to evaluate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    # Human-friendly titles for corruption types
    title_map = {
        'gaussian_noise': 'Noise (Gaussian)',
        'defocus_blur': 'Blur (Defocus)',
        'snow': 'Weather (Snow)',
        'jpeg_compression': 'Digital (Jpeg)',
    }

    for ctype in tqdm(args.corruptions, desc="Corruptions"):
        # Load the specified corruption at given severity
        x_test, y_test = load_cifar10c(args.num_examples, args.severity, args.data_dir, False, [ctype])
        # Resize to 384x384 for ViT-B/16-384
        x_test = F.interpolate(x_test, size=(384, 384), mode='bilinear', align_corners=False)

        start, stop, step = args.progression
        # Build ratios including the stop value
        ratios_list = list(range(start, stop + 1, max(step, 1)))
        if ratios_list[-1] != stop:
            ratios_list.append(stop)

        ratios, errors, entropies = evaluate_phase_mix_trend(
            model, x_test, y_test, device,
            batch_size=args.batch_size,
            ratios=ratios_list,
            phase_alpha=args.phase_alpha,
            save_mix_examples=args.save_mix_examples,
            mix_example_levels=args.mix_example_levels,
            figs_dir=args.figs_dir,
            example_tag=ctype,
        )

        title = f"{title_map.get(ctype, ctype)}, alpha={args.phase_alpha}"
        out_name = f'{ctype}.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(ratios, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
