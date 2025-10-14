import os
import math
import logging
from copy import deepcopy
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse helpers from EntREM implementation
import entrem

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

logger = logging.getLogger(__name__)


def copy_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer,
                             model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


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


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


class ThreeViewREM(nn.Module):
    """
    Three-view REM variant using ranked patch entropies:
      - v0: original image
      - v1: mask k random patches (k = round(m * P^2))
      - v2: mask k highest-entropy patches

    Losses:
      - MCL (mask consistency): cross-entropy consistency among views
      - ERL (entropy ranking): enforce entropy ordering across views
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 steps: int = 1,
                 episodic: bool = False,
                 m: float = 0.1,
                 patch_size: int = 16,
                 num_bins: int = 32,
                 use_color_entropy: bool = False,
                 use_mcl: bool = True,
                 use_erl: bool = True,
                 mcl_distance: str = 'CE',
                 plot_enable: bool = False,
                 plot_ema: float = 0.9,
                 plot_dir: str = "cifar/plots/FreqREM/Loss",
                 plot_filename: str = "losses.png",
                 ): 
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ThreeViewREM requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        # Masking controls
        self.m = float(m)  # fraction in [0,1]
        self.patch_size = int(patch_size)
        self.num_bins = int(num_bins)
        self.use_color_entropy = bool(use_color_entropy)

        # Loss toggles / config
        self.use_mcl = bool(use_mcl)
        self.use_erl = bool(use_erl)
        md = str(mcl_distance).upper()
        if md == 'COS':
            md = 'COSINE'
        if md == 'REVERSE_KL':
            md = 'RKL'
        self.mcl_distance = md

        # Plotting
        self.plot_enable = bool(plot_enable)
        self.plot_ema = float(plot_ema)
        self.plot_dir = str(plot_dir)
        self.plot_filename = str(plot_filename)
        self._loss_hist = {"MCL": [], "ERL": []}
        self._loss_ema_last = {}
        self._loss_ema_series = {"MCL": [], "ERL": []}

        # Entropy util (reuse implementation from entrem)
        self.entropy = entrem.Entropy()

        # Margin scale for ERL; set default 0 (can be set from outside if needed)
        self.margin_mult = 0.0
        self.lamb = 1.0

    def set_hyper(self, lamb: float = 1.0, margin: float = 0.0):
        self.lamb = float(lamb)
        self.margin_mult = float(margin)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        # Clear histories on reset for plotting
        self._loss_hist = {"MCL": [], "ERL": []}
        self._loss_ema_last = {}
        self._loss_ema_series = {"MCL": [], "ERL": []}

    @torch.enable_grad()
    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()
        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x, self.optimizer)
        return out

    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"Input size {(H, W)} must be divisible by patch_size {self.patch_size}")
        P = H // self.patch_size
        Np = P * P
        k = max(0, int(round(self.m * Np)))

        # Build entropy map per image
        xin = x if self.use_color_entropy else entrem.rgb_to_grayscale(x)
        xin = xin.clamp(0.0, 1.0)
        ent_map = entrem.compute_patch_entropy_map(xin, patches_per_side=P, num_bins=self.num_bins)  # [B,P,P]

        outputs_list: List[torch.Tensor] = []
        self.model.eval()

        # View 0: unmasked
        out0 = self.model(x, return_attn=False)
        outputs_list.append(out0)

        # Views 1 and 2: with masking when k>0
        if k == 0:
            out1 = out0
            out2 = out0
        else:
            xb1 = x.clone()
            xb2 = x.clone()
            for bi in range(B):
                scores_b = ent_map[bi].flatten()  # [Np]
                # Random indices
                idxs_rand = torch.randperm(Np)[:k]
                # Highest-entropy indices
                _, idxs_high = torch.topk(scores_b, k, largest=True)
                mask_rand = build_patch_selection_mask(H, W, P, idxs_rand).to(x.device)
                mask_high = build_patch_selection_mask(H, W, P, idxs_high.to(torch.long)).to(x.device)
                xb1[bi] = xb1[bi] * (1.0 - mask_rand.unsqueeze(0))
                xb2[bi] = xb2[bi] * (1.0 - mask_high.unsqueeze(0))
            out1 = self.model(xb1, return_attn=False)
            out2 = self.model(xb2, return_attn=False)
        outputs_list.append(out1)
        outputs_list.append(out2)
        self.model.train()

        # Compute losses
        total_loss = 0.0
        mcl_val = None
        erl_val = None

        if self.use_mcl:
            mcl_accum = 0.0
            # i in {1,2}; compare to view0 and to each other
            for i in range(1, len(outputs_list)):
                # anchor = less-masked (view0); pred = current view i
                mcl_accum = mcl_accum + self._prob_distance(outputs_list[0].detach(), outputs_list[i], self.mcl_distance).mean()
                for j in range(1, i):
                    # anchor = less-masked view j; pred = more-masked view i
                    mcl_accum = mcl_accum + self._prob_distance(outputs_list[j].detach(), outputs_list[i], self.mcl_distance).mean()
            total_loss = total_loss + mcl_accum
            mcl_val = mcl_accum.detach()

        if self.use_erl:
            entropys = [self.entropy(o) for o in outputs_list]
            margin = self.margin_mult * math.log(outputs_list[0].shape[-1])
            erl_accum = 0.0
            for i in range(len(outputs_list)):
                for j in range(i + 1, len(outputs_list)):
                    erl_accum = erl_accum + (F.relu(entropys[i] - entropys[j].detach() + margin)).mean()
            total_loss = total_loss + self.lamb * erl_accum
            erl_val = (self.lamb * erl_accum).detach()

        # Record EMA histories
        self._update_loss_hist(mcl_val, erl_val)

        # Optimize
        loss = total_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Return unmasked prediction
        return outputs_list[0]

    def _prob_distance(self, logits_anchor: torch.Tensor, logits_pred: torch.Tensor, metric: str) -> torch.Tensor:
        """Compute distance between probability distributions derived from logits.
        anchor is detached target (no grad), pred carries gradient.
        Returns per-sample distance (shape [B]).
        """
        eps = 1e-8
        P = logits_anchor.softmax(dim=1)
        Q = logits_pred.softmax(dim=1)
        if metric == 'CE':
            # Cross-entropy H(P, Q) = - sum P * log Q
            return -(P * (Q + eps).log()).sum(dim=1)
        elif metric == 'KL':
            # D_KL(P || Q) = sum P * (log P - log Q)
            return (P * ((P + eps).log() - (Q + eps).log())).sum(dim=1)
        elif metric == 'RKL':
            # Reverse KL: D_KL(Q || P)
            return (Q * ((Q + eps).log() - (P + eps).log())).sum(dim=1)
        elif metric == 'JS':
            # Jensen-Shannon divergence
            M = 0.5 * (P + Q)
            d1 = (P * ((P + eps).log() - (M + eps).log())).sum(dim=1)
            d2 = (Q * ((Q + eps).log() - (M + eps).log())).sum(dim=1)
            return 0.5 * (d1 + d2)
        elif metric == 'COSINE':
            Pn = F.normalize(P, p=2, dim=1)
            Qn = F.normalize(Q, p=2, dim=1)
            return 1.0 - (Pn * Qn).sum(dim=1)
        elif metric == 'L1':
            return torch.abs(P - Q).mean(dim=1)
        elif metric == 'L2':
            return torch.pow(P - Q, 2).mean(dim=1)
        elif metric == 'HELLINGER':
            # Hellinger distance (squared form): 0.5 * sum (sqrt(P) - sqrt(Q))^2
            sP = torch.sqrt(torch.clamp(P, min=0.0))
            sQ = torch.sqrt(torch.clamp(Q, min=0.0))
            return 0.5 * torch.pow(sP - sQ, 2).sum(dim=1)
        else:
            # default to CE
            return -(P * (Q + eps).log()).sum(dim=1)

    def _update_loss_hist(self, mcl_val: Optional[torch.Tensor], erl_val: Optional[torch.Tensor]):
        if not self.plot_enable:
            return
        def ema_update(name: str, val: Optional[torch.Tensor]):
            if val is None:
                return
            v = float(val.detach().cpu().item())
            self._loss_hist[name].append(v)
            last = self._loss_ema_last.get(name, v)
            ema = self.plot_ema * last + (1.0 - self.plot_ema) * v
            self._loss_ema_last[name] = ema
            self._loss_ema_series[name].append(ema)
        ema_update("MCL", mcl_val)
        ema_update("ERL", erl_val)

    def save_loss_plot(self):
        if not self.plot_enable:
            return
        if not HAVE_MPL:
            logger.warning("Matplotlib not available; skipping loss plot saving.")
            return
        os.makedirs(self.plot_dir, exist_ok=True)
        # Allow placeholders in filename: {m}, {patch_size}, {ps}, {mcl_distance}
        try:
            formatted_fname = self.plot_filename.format(
                m=self.m,
                patch_size=self.patch_size,
                ps=self.patch_size,
                mcl_distance=self.mcl_distance
            )
        except Exception:
            base, ext = (self.plot_filename.rsplit('.', 1) + ["png"])[:2]
            formatted_fname = f"{base}_m{self.m}_ps{self.patch_size}_dist{self.mcl_distance}.{ext}"
        path = os.path.join(self.plot_dir, formatted_fname)
        plt.figure(figsize=(10, 6))
        any_line = False
        # Only plot enabled losses with data
        series_items = []
        if self.use_mcl and len(self._loss_ema_series.get("MCL", [])) > 0:
            series_items.append(("MCL", self._loss_ema_series["MCL"]))
        if self.use_erl and len(self._loss_ema_series.get("ERL", [])) > 0:
            series_items.append(("ERL", self._loss_ema_series["ERL"]))
        for name, series in series_items:
            plt.plot(series, label=name, linewidth=2)
            any_line = True
        if not any_line:
            logger.info("No enabled losses with data to plot; skipping save.")
            plt.close()
            return
        plt.title("EMA of Loss Components")
        plt.xlabel("Step")
        plt.ylabel("Loss (EMA)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
