from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Entropy(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between current logits and a detached target distribution."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


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


def collect_params(model: nn.Module):
    """Collect trainable parameters similar to REM's policy (skip late ViT blocks and norms)."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
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


class EntREM(nn.Module):
    """
    Entropy-based REM variant: instead of masking tokens via attention, we compute a patchwise
    entropy map on the input image, determine a weighted centroid from the top fraction of high-entropy
    patches, and apply a single grid-aligned square mask of the corresponding area centered at this point.
    We then compute the REM losses across masking levels and update the model.
    """
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 steps: int = 1, episodic: bool = False,
                 m: float = 0.1, n: int = 3, lamb: float = 1.0, margin: float = 0.0,
                 patch_size: int = 16, num_bins: int = 32,
                 use_color_entropy: bool = False, entropy_weight_power: float = 2.0):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "EntREM requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.m = m
        self.n = n
        self.mn = [i * self.m for i in range(self.n)]
        self.lamb = lamb
        self.margin = margin

        self.entropy = Entropy()

        # Entropy-masking params
        self.patch_size = patch_size
        self.num_bins = num_bins
        self.use_color_entropy = use_color_entropy
        self.entropy_weight_power = entropy_weight_power

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

    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Ensure size is divisible by patch_size
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input size {(H, W)} must be divisible by patch_size {self.patch_size}")
        patches_per_side = H // self.patch_size

        # Precompute entropy map for each image (grayscale or color averaged)
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
        for m in self.mn:
            if m == 0.0:
                out = self.model(x, return_attn=False)
                outputs_list.append(out)
            else:
                mfrac = m  # already fraction in [0,1]
                xb_masked = x.clone()
                Ph, Pw = ent_norm.shape[1], ent_norm.shape[2]
                Np = Ph * Pw
                k = max(1, int(round(mfrac * Np)))
                patch_h = H // Ph
                patch_w = W // Pw
                for bi in range(B):
                    scores_b = ent_norm[bi].flatten()  # [Np]
                    vals, idxs = torch.topk(scores_b, k, largest=True)
                    rows = (idxs // Pw).float()
                    cols = (idxs % Pw).float()
                    eps_w = 1e-8
                    w = (vals.float() + eps_w) ** float(self.entropy_weight_power)
                    r_bar = (rows * w).sum() / w.sum()
                    c_bar = (cols * w).sum() / w.sum()
                    cy = int(torch.round((r_bar + 0.5) * patch_h).item())
                    cx = int(torch.round((c_bar + 0.5) * patch_w).item())
                    total_area = int(round(mfrac * H * W))
                    side = int(round(math.sqrt(max(total_area, 1))))
                    side = max(patch_h, (side // patch_h) * patch_h)
                    side = min(side, min(H, W))
                    mask_bw = build_centered_square_mask(H, W, side, cy, cx).to(xb_masked.device)
                    xb_masked[bi] = xb_masked[bi] * (1.0 - mask_bw.unsqueeze(0))
                out = self.model(xb_masked, return_attn=False)
                outputs_list.append(out)
        self.model.train()

        # Compute REM losses across masking levels
        loss = 0.0
        for i in range(1, len(self.mn)):
            loss = loss + softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
            for j in range(1, i):
                loss = loss + softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()

        entropys = [self.entropy(o) for o in outputs_list]
        margin = self.margin * math.log(outputs_list[0].shape[-1])
        lossn = 0.0
        for i in range(len(self.mn)):
            for j in range(i + 1, len(self.mn)):
                lossn = lossn + (F.relu(entropys[i] - entropys[j].detach() + margin)).mean()

        loss = loss + self.lamb * lossn
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Return unmasked prediction
        return outputs_list[0]
