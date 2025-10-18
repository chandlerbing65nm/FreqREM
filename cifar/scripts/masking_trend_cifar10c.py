#!/usr/bin/env python3
import os
import sys
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
                           ratios: list = None,
                           save_mask_examples: int = 0,
                           mask_example_levels: list = None,
                           mask_figs_dir: str = None,
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
    # Defaults for saving examples
    if mask_example_levels is None:
        mask_example_levels = [0, 10, 20]
    saved_examples = 0

    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end].to(device, non_blocking=True)
            yb = y[start:end].to(device, non_blocking=True)
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

            # Optionally save masked example figures with attention maps for a few samples
            if save_mask_examples > 0 and saved_examples < save_mask_examples and mask_figs_dir is not None:
                os.makedirs(mask_figs_dir, exist_ok=True)
                B = xb.shape[0]
                to_take = min(save_mask_examples - saved_examples, B)
                # Use CLI-provided example levels for visualization
                viz_levels = mask_example_levels
                for bi in range(to_take):
                    # Work on a single sample
                    x1_cpu = xb[bi].detach().cpu().clone()  # [C,H,W], in [0,1]
                    x1b = xb[bi:bi+1]  # keep on device for forward calls
                    # Precompute helpers for masking
                    patches_per_side = 24  # 384 / 16
                    patch_h = x1_cpu.shape[1] // patches_per_side
                    patch_w = x1_cpu.shape[2] // patches_per_side
                    attn1 = attn_score[bi:bi+1]  # [1, 576], on device

                    imgs_row = []     # top row: input/masked images
                    attn_row = []     # bottom row: attention heatmaps
                    labels = []

                    for lv in viz_levels:
                        m = float(lv) / 100.0
                        num_keep = int(tokens * (1.0 - m))
                        if num_keep > 0:
                            keep_idx = torch.topk(attn1, num_keep, largest=False).indices[0]  # [num_keep] (device)
                        else:
                            keep_idx = attn1[0][:0]
                        keep_set = set(keep_idx.detach().cpu().tolist())

                        # Build masked image by zeroing masked patches (CPU tensor for plotting)
                        xm = x1_cpu.clone()
                        for p in range(tokens):
                            if p not in keep_set:
                                row = p // patches_per_side
                                col = p % patches_per_side
                                r0 = row * patch_h
                                r1 = r0 + patch_h
                                c0 = col * patch_w
                                c1 = c0 + patch_w
                                xm[:, r0:r1, c0:c1] = 0.0
                        imgs_row.append(xm)
                        labels.append(f"{lv}%")

                        # Build attention map for this masking level
                        if lv == 0:
                            # Use unmasked attention from the batch forward
                            att_vec = attn1[0]  # [576]
                            full_map = att_vec.detach().cpu().reshape(patches_per_side, patches_per_side)
                        else:
                            # Re-run model to get attention under masked configuration
                            with torch.no_grad():
                                logits_m, attn_m = model(x1b, len_keep=keep_idx.unsqueeze(0), return_attn=True)
                            att_vec_kept = attn_m.mean(dim=1)[0, 0, 1:]  # [num_keep]
                            # Scatter kept attentions back into a 576-long vector, masked tokens as 0
                            full_vec = torch.zeros(tokens, device=att_vec_kept.device)
                            if num_keep > 0:
                                full_vec[keep_idx] = att_vec_kept
                            full_map = full_vec.detach().cpu().reshape(patches_per_side, patches_per_side)

                        # Normalize attention map for visualization (min-max over kept tokens; zeros remain 0)
                        amap = full_map
                        if amap.max().item() > 1e-12:
                            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-12)
                        attn_row.append(amap)

                    # Plot a 2xK panel (top: images, bottom: attention heatmaps)
                    K = len(viz_levels)
                    fig, axes = plt.subplots(2, K, figsize=(3*K, 6))
                    # Ensure axes is 2D array even if K==1
                    if K == 1:
                        axes = axes.reshape(2, 1)

                    # Top row: images
                    for j in range(K):
                        axes[0, j].imshow(imgs_row[j].permute(1, 2, 0).numpy())
                        axes[0, j].set_title(labels[j])
                        axes[0, j].axis('off')

                    # Bottom row: attention heatmaps
                    for j in range(K):
                        axes[1, j].imshow(attn_row[j].numpy(), cmap='magma', vmin=0.0, vmax=1.0)
                        axes[1, j].axis('off')

                    tag = example_tag or "examples"
                    out_ex = os.path.join(mask_figs_dir, f"{tag}.png")
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
            ratios=ratios_list,
            save_mask_examples=args.save_mask_examples,
            mask_example_levels=args.mask_example_levels,
            mask_figs_dir=args.mask_figs_dir,
            example_tag=ctype,
        )

        title = title_map.get(ctype, ctype)
        out_name = f'{ctype}_masking_trend.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_trend(ratios, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
