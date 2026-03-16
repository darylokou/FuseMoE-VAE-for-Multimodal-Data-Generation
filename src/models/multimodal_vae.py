from typing import Dict, Optional

import torch
import torch.nn as nn

from losses.kl import kl_standard_normal


class MultimodalVAE(nn.Module):
    """Minimal multimodal VAE wrapper for the MVP pipeline."""

    def __init__(
        self,
        ts_encoder: nn.Module,
        tabular_encoder: nn.Module,
        fusion: nn.Module,
        posterior: nn.Module,
        tabular_decoder: nn.Module,
        ts_decoder: Optional[nn.Module] = None,
        kl_reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.ts_encoder = ts_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion = fusion
        self.posterior = posterior
        self.tabular_decoder = tabular_decoder
        self.ts_decoder = ts_decoder
        self.kl_reduction = kl_reduction

    def forward(self, batch: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, object]:
        ts_out = self.ts_encoder(
            values=batch["ts_values"],
            mask=batch["ts_mask"],
            times=batch["ts_times"],
        )
        tab_out = self.tabular_encoder(
            x_num=batch.get("tab_num"),
            x_cat=batch.get("tab_cat"),
        )
        fused = self.fusion({
            "ts": ts_out["pooled"],
            "tab": tab_out["pooled"],
        })
        post = self.posterior(fused["pooled"], deterministic=deterministic)

        tab_decoded = self.tabular_decoder(post["z"])
        ts_decoded = self.ts_decoder(post["z"]) if self.ts_decoder is not None else None

        kl = kl_standard_normal(post["mu"], post["logvar"], reduction=self.kl_reduction)

        return {
            "ts_encoder": ts_out,
            "tabular_encoder": tab_out,
            "fusion": fused,
            "posterior": post,
            "tabular_decoder": tab_decoded,
            "ts_decoder": ts_decoded,
            "losses": {
                "kl": kl,
                "balance_loss": fused["balance_loss"],
            },
        }


__all__ = ["MultimodalVAE"]
