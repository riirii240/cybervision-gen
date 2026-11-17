"""Gradient penalty de Gulrajani et al. pour données tabulaires conditionnelles."""
from __future__ import annotations

import torch


def gradient_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    target_norm: float = 1.0,
) -> torch.Tensor:
    """Calcule E[(||grad_x C(x,y)||2 - 1)^2] pour des paires de même classe.

    ``fake`` n'est pas détaché ici : la fonction reste générique. Pendant la mise à
    jour du critique, l'appelant génère toutefois ``fake`` sous ``torch.no_grad()``.
    """
    if real.shape != fake.shape or real.ndim != 2:
        raise ValueError("real et fake doivent partager une forme tabulaire (N, F).")
    if labels.numel() != real.shape[0]:
        raise ValueError("Un label est requis par exemple.")
    alpha = torch.rand(real.shape[0], 1, device=real.device, dtype=real.dtype)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norms = gradients.flatten(start_dim=1).norm(2, dim=1)
    return ((norms - target_norm) ** 2).mean()
