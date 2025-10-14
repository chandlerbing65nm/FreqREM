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

# Ensure local robustbench fork is importable when running from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CIFAR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if CIFAR_DIR not in sys.path:
    sys.path.insert(0, CIFAR_DIR)

from robustbench.data import load_cifar10c
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model


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


def rgb_to_grayscale(x: torch.Tensor) -> torch.Tensor:
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    return (0.299 * r + 0.587 * g + 0.114 * b)


def compute_patch_entropy_map(x_img: torch.Tensor,
                              patches_per_side: int,
                              num_bins: int) -> torch.Tensor:
    """
    Compute Shannon entropy per patch over a patches_per_side x patches_per_side grid.
    x_img: [B,C,H,W] in [0,1], C can be 1 (grayscale) or 3 (RGB)
    Returns: [B, patches_per_side, patches_per_side]
    """
    B, C, H, W = x_img.shape
    assert H % patches_per_side == 0 and W % patches_per_side == 0, "H and W must be divisible by patches_per_side"
    patch_h = H // patches_per_side
    patch_w = W // patches_per_side

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

    probs = counts / float(patch_size)
    eps = 1e-8
    ent = -(probs * (probs + eps).log()).sum(dim=1)

    ent = ent.view(B, C, Ph, Pw)
    ent = ent.mean(dim=1)  # [B,Ph,Pw]
    return ent


def build_patch_selection_mask(H: int, W: int, patches_per_side: int, selected_idxs: torch.Tensor) -> torch.Tensor:
    """
    Build a binary mask [H,W] where patches at indices in selected_idxs (flat indices over Ph*Pw) are set to 1.
    selected_idxs: 1D LongTensor of length K containing indices in [0, Ph*Pw-1].
    """
    patch_h = H // patches_per_side
    patch_w = W // patches_per_side
    Ph = Pw = patches_per_side
    mask = torch.zeros((H, W), dtype=torch.float32)
    for idx in selected_idxs.tolist():
        pr = idx // Pw
        pc = idx % Pw
        y0 = pr * patch_h
        x0 = pc * patch_w
        mask[y0:y0 + patch_h, x0:x0 + patch_w] = 1.0
    return mask


def evaluate_ranked_entropy_views(model: torch.nn.Module,
                                  x: torch.Tensor,
                                  y: torch.Tensor,
                                  device: torch.device,
                                  batch_size: int,
                                  patches_per_side: int,
                                  num_bins: int,
                                  use_color_entropy: bool,
                                  save_examples: int = 0,
                                  figs_dir: str = None,
                                  example_class_idx: int = None,
                                  example_tag: str = "",
                                  rng: Optional[torch.Generator] = None,
                                  ) -> Tuple[List[str], List[float], List[float]]:
    """
    Evaluate three views:
      - view0: unmasked
      - view1_rand: mask A% random patches
      - view2_high: mask A% highest-entropy patches
    Returns: (labels, errors, entropies)
    """
    model.eval()

    labels = ["view0", "view1_rand", "view2_high"]
    correct = [0, 0, 0]
    ent_sums = [0.0, 0.0, 0.0]
    total = 0

    saved_examples = 0
    with torch.no_grad():
        N = x.shape[0]
        for start in tqdm(range(0, N, batch_size), total=(N + batch_size - 1)//batch_size, desc="Batches", leave=False):
            end = min(start + batch_size, N)
            xb = x[start:end]
            yb = y[start:end].to(device, non_blocking=True)

            # Resize per batch on device to reduce memory
            xb = xb.to(device, non_blocking=True)
            xb = F.interpolate(xb, size=(384, 384), mode='bilinear', align_corners=False)
            B, C, H, W = xb.shape

            # Build entropy map
            xin = xb if use_color_entropy else rgb_to_grayscale(xb)
            xin = xin.clamp(0.0, 1.0)
            ent_map = compute_patch_entropy_map(xin, patches_per_side=patches_per_side, num_bins=num_bins)  # [B,Ph,Pw]

            # Flatten for ranking
            Ph = Pw = patches_per_side
            Np = Ph * Pw

            # view0: no masking
            logits0 = model(xb, return_attn=False)
            pred0 = logits0.argmax(dim=1)
            correct[0] += (pred0 == yb).sum().item()
            ent_sums[0] += entropy_from_logits(logits0).sum().item()

            # For views 1 and 2: create per-image patch masks and apply
            # Compute number of patches to mask based on percentage argument set outside this function
            # We'll pass A via a closure or outer variable; to keep the function self-contained, we'll read from global _MASK_PERCENT set by main
            m = max(0.0, min(1.0, _MASK_PERCENT))
            k = max(1, int(round(m * Np))) if m > 0 else 0

            if k == 0:
                # Nothing to mask: identical to view0
                logits1 = logits0
                logits2 = logits0
            else:
                xb1 = xb.clone()
                xb2 = xb.clone()
                rand_idxs_list: List[Optional[torch.Tensor]] = [None] * B
                high_idxs_list: List[Optional[torch.Tensor]] = [None] * B
                for bi in range(B):
                    scores_b = ent_map[bi].flatten()  # [Np]
                    # random indices
                    if rng is not None:
                        perm = torch.randperm(Np, generator=rng)
                    else:
                        perm = torch.randperm(Np)
                    idxs_rand = perm[:k]
                    # highest entropy indices
                    _, idxs_high = torch.topk(scores_b, k, largest=True)

                    rand_idxs_list[bi] = idxs_rand.to(torch.long)
                    high_idxs_list[bi] = idxs_high.to(torch.long)

                    mask_rand = build_patch_selection_mask(H, W, patches_per_side, rand_idxs_list[bi]).to(device)
                    mask_high = build_patch_selection_mask(H, W, patches_per_side, high_idxs_list[bi]).to(device)

                    xb1[bi] = xb1[bi] * (1.0 - mask_rand.unsqueeze(0))
                    xb2[bi] = xb2[bi] * (1.0 - mask_high.unsqueeze(0))

                logits1 = model(xb1, return_attn=False)
                logits2 = model(xb2, return_attn=False)

            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct[1] += (pred1 == yb).sum().item()
            correct[2] += (pred2 == yb).sum().item()
            ent_sums[1] += entropy_from_logits(logits1).sum().item()
            ent_sums[2] += entropy_from_logits(logits2).sum().item()

            total += (end - start)

            # Save example figures if requested
            if save_examples > 0 and saved_examples < save_examples and figs_dir is not None:
                os.makedirs(figs_dir, exist_ok=True)
                # Build CPU copies for visualization
                xb_cpu = xb.detach().cpu()
                # Rebuild masks for the saved views per image to get the binary maps
                for bi in range(B):
                    if saved_examples >= save_examples:
                        break
                    if example_class_idx is not None and int(yb[bi].detach().cpu().item()) != example_class_idx:
                        continue
                    # Prepare three images: view0, view1_rand, view2_high
                    if k > 0:
                        # Use the same selections as used for forward pass when available; otherwise recompute
                        if 'rand_idxs_list' in locals() and rand_idxs_list[bi] is not None:
                            idxs_rand = rand_idxs_list[bi]
                        else:
                            if rng is not None:
                                perm = torch.randperm(Np, generator=rng)
                            else:
                                perm = torch.randperm(Np)
                            idxs_rand = perm[:k]
                        if 'high_idxs_list' in locals() and high_idxs_list[bi] is not None:
                            idxs_high = high_idxs_list[bi]
                        else:
                            scores_b = ent_map[bi].flatten()
                            _, idxs_high = torch.topk(scores_b, k, largest=True)
                        mask_rand = build_patch_selection_mask(H, W, patches_per_side, idxs_rand.to(torch.long))
                        mask_high = build_patch_selection_mask(H, W, patches_per_side, idxs_high.to(torch.long))
                    else:
                        mask_rand = torch.zeros((H, W), dtype=torch.float32)
                        mask_high = torch.zeros((H, W), dtype=torch.float32)

                    img0 = xb_cpu[bi]
                    img1 = xb_cpu[bi] * (1.0 - mask_rand.unsqueeze(0))
                    img2 = xb_cpu[bi] * (1.0 - mask_high.unsqueeze(0))

                    # Also compute attention heatmaps for each masked view
                    attns = []
                    for mask in [torch.zeros((H, W), dtype=torch.float32, device=device),
                                 mask_rand.to(device), mask_high.to(device)]:
                        x_masked_b = xb[bi:bi+1] * (1.0 - mask.unsqueeze(0).unsqueeze(0))
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
                        attns.append(attn_norm.detach().cpu())

                    images = [img0, img1, img2]
                    labels_local = ["view0", "view1_rand", "view2_high"]

                    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
                    for j in range(3):
                        axes[0, j].imshow(images[j].permute(1, 2, 0).numpy())
                        axes[0, j].set_title(labels_local[j])
                        axes[0, j].axis('off')
                    for j in range(3):
                        axes[1, j].imshow(attns[j].numpy(), cmap='inferno')
                        axes[1, j].axis('off')
                    class_idx = int(yb[bi].detach().cpu().item())
                    class_name = CIFAR10_CLASSES[class_idx] if 0 <= class_idx < len(CIFAR10_CLASSES) else f"class{class_idx}"
                    out_name = example_tag or "examples"
                    out_fp = os.path.join(figs_dir, f"{out_name}_idx{start+bi:05d}_{class_name}.png")
                    fig.tight_layout()
                    fig.savefig(out_fp, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    saved_examples += 1

    errors = [1.0 - (c / total) for c in correct]
    entropies = [h / total for h in ent_sums]
    return labels, errors, entropies


def plot_three_view_trend(labels: List[str], errors: List[float], entropies: List[float], title: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    xs = list(range(len(labels)))

    errors_pct = [e * 100.0 for e in errors]
    line_err = ax1.plot(xs, errors_pct, marker='o', color='tab:red', linewidth=3, label='Error (%)')[0]
    ax1.set_xlabel('Views')
    ax1.set_ylabel('Error (%)', color='tab:red')
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 100.0)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    line_ent = ax2.plot(xs, entropies, marker='s', color='tab:blue', linewidth=3, label='Entropy')[0]
    ax2.set_ylabel('Entropy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(title)
    lines = [line_err, line_ent]
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc='best')
    fig.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Ranked-entropy three-view masking on CIFAR-10-C for source model (no adaptation).')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to CIFAR-10-C data directory')
    parser.add_argument('--ckpt_dir', type=str, default='/users/doloriel/work/Repo/FreqREM/ckpt', help='Checkpoint directory')
    parser.add_argument('--checkpoint', type=str, default='/users/doloriel/work/Repo/FreqREM/ckpt/vit_base_384_cifar10.t7', help='Path to ViT-Base 384 checkpoint')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_examples', type=int, default=10000, help='Number of examples per corruption')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=os.path.join(CIFAR_DIR, 'plots'))
    parser.add_argument('--patches_per_side', type=int, default=24, help='Number of patches per side (on 384x384)')
    parser.add_argument('--entropy_bins', type=int, default=16, help='Histogram bins for patch entropy')
    parser.add_argument('--use_color_entropy', action='store_true', help='Compute entropy on RGB channels (avg) instead of grayscale')
    parser.add_argument('--mask_percent', type=float, default=20.0, help='A: percentage of patches to mask in views 1 and 2')
    parser.add_argument('--corruptions', type=str, nargs='+', default=['gaussian_noise', 'defocus_blur', 'snow', 'jpeg_compression'], help='List of CIFAR-10-C corruptions to evaluate')
    parser.add_argument('--save_examples', type=int, default=0, help='Number of samples to save per corruption (0 disables)')
    parser.add_argument('--figs_dir', type=str, default=None, help='Directory to save example view PNGs')
    parser.add_argument('--example_class', type=str, default=None, help='If set, only save examples for this class (name or index 0-9)')
    parser.add_argument('--random_seed', type=int, default=None, help='Optional random seed for reproducible random masking')
    args = parser.parse_args()

    if args.patches_per_side <= 0:
        raise ValueError('--patches_per_side must be positive')
    if not (0.0 <= args.mask_percent <= 100.0):
        raise ValueError('--mask_percent must be in [0, 100]')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global _MASK_PERCENT
    _MASK_PERCENT = float(args.mask_percent) / 100.0

    model = build_source_model(args.ckpt_dir, args.checkpoint, device)

    # Optional RNG for reproducible random masking
    rng = None
    if args.random_seed is not None:
        rng = torch.Generator(device='cpu')
        rng.manual_seed(int(args.random_seed))

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

    for ctype in tqdm(args.corruptions, desc='Corruptions'):
        x_test, y_test = load_cifar10c(args.num_examples, args.severity, args.data_dir, False, [ctype])
        # Do not resize here (avoid huge host memory); we resize per-batch on device in evaluation

        labels, errors, entropies = evaluate_ranked_entropy_views(
            model, x_test, y_test, device,
            batch_size=args.batch_size,
            patches_per_side=args.patches_per_side,
            num_bins=args.entropy_bins,
            use_color_entropy=args.use_color_entropy,
            save_examples=args.save_examples,
            figs_dir=args.figs_dir,
            example_class_idx=target_class_idx,
            example_tag=ctype,
            rng=rng,
        )

        title = f'{ctype}'
        out_name = f'{ctype}.png'
        out_path = os.path.join(args.out_dir, out_name)
        plot_three_view_trend(labels, errors, entropies, title, out_path)
        print(f'Saved plot: {out_path}')


if __name__ == '__main__':
    main()
