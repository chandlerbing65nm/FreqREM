import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rem import (
    configure_model,
    collect_params,
    copy_model_and_optimizer,
    load_model_and_optimizer,
    Entropy,
    softmax_entropy,
)


class REMPhase(nn.Module):
    """
    Test-time adaptation with progressive per-channel phase distortion variants
    (0 channels, then 1, then 1+2, then 1+2+3), instead of patch-token masking.
    Uses the same MCL and ERL losses as REM.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
        levels=(0.0, 0.25, 0.30),  # deprecated: kept for backward-compat (ignored)
        lamb: float = 1.0,
        margin: float = 0.0,
        phase_seed: int = None,
        alpha: float = 0.45,
        channel_order=(0, 1, 2),
        channel_steps=(0, 1, 2, 3),
        use_mcl: bool = True,
        use_erl: bool = True,
        consistency_mode: str = "mcl",
        cwal_threshold: float = 0.7,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "REMPhase requires >= 1 step(s)"
        self.episodic = episodic

        self.model_state, self.optimizer_state, _, _ = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # Per-channel progression configuration
        self.alpha = float(alpha)
        self.channel_order = tuple(int(c) for c in channel_order)
        self.channel_steps = tuple(int(s) for s in channel_steps)

        self.lamb = lamb
        self.margin = margin

        self.entropy = Entropy()
        self.phase_seed = phase_seed
        self.use_mcl = bool(use_mcl)
        self.use_erl = bool(use_erl)
        self.consistency_mode = str(consistency_mode).lower()
        assert self.consistency_mode in ["mcl", "cwal"], "consistency_mode must be 'mcl' or 'cwal'"
        self.cwal_threshold = float(cwal_threshold)

    def forward(self, x: torch.Tensor):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    @torch.enable_grad()
    def forward_and_adapt(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Unmodified forward (builds the graph for gradients)
        outputs = self.model(x, return_attn=False)

        # Build phase-distorted variants with model in eval mode (stable stats) but keep autograd ON
        self.model.eval()
        # Fixed random unit-phase target per batch (no gradient needed for FFT construction)
        B, C, H, W = x.shape
        if self.phase_seed is not None:
            torch.manual_seed(self.phase_seed)
        with torch.no_grad():
            psi = (torch.rand((B, C, H, W), device=x.device) * 2 * math.pi) - math.pi
            unit_rand = torch.polar(torch.ones_like(psi), psi)  # e^{i psi}

            # Precompute original FFT magnitude & phase-unit
            X = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
            mag = torch.abs(X)
            eps = 1e-8
            unit = X / (mag + eps)

        outputs_list = [outputs]
        eps = 1e-8
        # Generate variants by increasing the number of channels that receive phase distortion
        for s in self.channel_steps:
            if s == 0:
                continue  # already have the clean output as anchor
            # Build distorted image (no grad through FFT math)
            with torch.no_grad():
                new_unit = unit.clone()
                num = min(s, len(self.channel_order), C)
                for k in range(num):
                    ch = self.channel_order[k]
                    mixed = (1.0 - self.alpha) * unit[:, ch] + self.alpha * unit_rand[:, ch]
                    new_unit[:, ch] = mixed
                new_unit = new_unit / (new_unit.abs() + eps)
                Xp = mag * new_unit
                x_rec = torch.fft.ifft2(Xp, dim=(-2, -1), norm='ortho').real
                x_rec = x_rec.clamp(0.0, 1.0)
            # Forward WITH gradients on model parameters
            out_s = self.model(x_rec, return_attn=False)
            outputs_list.append(out_s)
        self.model.train()

        # Build total loss with optional terms
        total_loss = None
        if self.use_mcl:
            if self.consistency_mode == "mcl":
                mcl = 0.0
                for i in range(1, len(outputs_list)):
                    mcl += softmax_entropy(outputs_list[i], outputs_list[0].detach()).mean()
                    for j in range(1, i):
                        mcl += softmax_entropy(outputs_list[i], outputs_list[j].detach()).mean()
                total_loss = mcl if total_loss is None else (total_loss + mcl)
            elif self.consistency_mode == "cwal":
                # Class-Wise Confidence CWAL (CWC-CWAL)
                # Use per-class confidence threshold to mask classes in the easy view.
                eps = 1e-8
                p_easy = outputs_list[0].softmax(dim=1)              # [B,C]
                p_easy_detached = p_easy.detach()
                class_mask = (p_easy > self.cwal_threshold).float()   # [B,C]
                if class_mask.sum() > 0:
                    cwal_total = 0.0
                    for i in range(1, len(outputs_list)):
                        p_hard_log = outputs_list[i].log_softmax(dim=1)  # [B,C]
                        # KL per class: p_easy * (log p_easy - log p_hard)
                        kl_per_class = p_easy_detached * (p_easy_detached.log() - p_hard_log)
                        masked_kl = kl_per_class * class_mask
                        loss_i = masked_kl.sum() / (class_mask.sum() + eps)
                        cwal_total += loss_i
                    total_loss = cwal_total if total_loss is None else (total_loss + cwal_total)

        if self.use_erl:
            entropys = [self.entropy(out) for out in outputs_list]
            erl = 0.0
            margin = self.margin * math.log(outputs.shape[-1])
            for i in range(len(outputs_list)):
                for j in range(i + 1, len(outputs_list)):
                    erl += (F.relu(entropys[i] - entropys[j].detach() + margin)).mean()
            erl = self.lamb * erl
            total_loss = erl if total_loss is None else (total_loss + erl)

        # If both terms disabled, skip update
        if total_loss is None:
            return outputs

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return outputs
