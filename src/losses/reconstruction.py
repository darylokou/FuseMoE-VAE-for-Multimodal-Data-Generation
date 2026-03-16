from typing import Dict, Literal, Optional, Sequence

import torch
import torch.nn.functional as F


Reduction = Literal["mean", "sum"]


def tabular_reconstruction_loss(
    target_num: Optional[torch.Tensor],
    pred_num: Optional[torch.Tensor],
    target_cat: Optional[torch.Tensor],
    pred_cat_logits: Optional[Sequence[torch.Tensor]],
    reduction: Reduction = "mean",
    numeric_loss: Literal["mse", "l1"] = "mse",
) -> Dict[str, torch.Tensor]:
    """Compute reconstruction losses for mixed tabular data."""
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    device = None
    for tensor in (target_num, pred_num, target_cat):
        if tensor is not None:
            device = tensor.device
            break
    if device is None and pred_cat_logits:
        device = pred_cat_logits[0].device
    if device is None:
        raise ValueError("At least one target/prediction tensor must be provided")

    num_loss = torch.zeros((), device=device)
    cat_loss = torch.zeros((), device=device)

    if target_num is not None or pred_num is not None:
        if target_num is None or pred_num is None:
            raise ValueError("target_num and pred_num must either both be provided or both be None")
        if target_num.shape != pred_num.shape:
            raise ValueError(
                f"target_num and pred_num must have the same shape, got {tuple(target_num.shape)} and {tuple(pred_num.shape)}"
            )
        if target_num.dim() != 2:
            raise ValueError(f"target_num and pred_num must be 2D, got {tuple(target_num.shape)}")

        if numeric_loss == "mse":
            num_loss = F.mse_loss(pred_num, target_num, reduction=reduction)
        elif numeric_loss == "l1":
            num_loss = F.l1_loss(pred_num, target_num, reduction=reduction)
        else:
            raise ValueError(f"Unsupported numeric_loss: {numeric_loss}")

    if target_cat is not None or pred_cat_logits is not None:
        if target_cat is None or pred_cat_logits is None:
            raise ValueError("target_cat and pred_cat_logits must either both be provided or both be None")
        if target_cat.dim() != 2:
            raise ValueError(f"target_cat must be 2D, got {tuple(target_cat.shape)}")
        if len(pred_cat_logits) != target_cat.size(1):
            raise ValueError(
                f"Expected {target_cat.size(1)} categorical heads, got {len(pred_cat_logits)}"
            )

        losses = []
        for j, logits in enumerate(pred_cat_logits):
            if logits.dim() != 2:
                raise ValueError(f"Categorical logits at index {j} must be 2D, got {tuple(logits.shape)}")
            if logits.size(0) != target_cat.size(0):
                raise ValueError(
                    f"Batch size mismatch for categorical head {j}: got {logits.size(0)} and {target_cat.size(0)}"
                )
            losses.append(F.cross_entropy(logits, target_cat[:, j], reduction=reduction))

        if losses:
            cat_loss = sum(losses)

    total = num_loss + cat_loss
    return {
        "num_loss": num_loss,
        "cat_loss": cat_loss,
        "total": total,
    }


__all__ = ["tabular_reconstruction_loss"]
