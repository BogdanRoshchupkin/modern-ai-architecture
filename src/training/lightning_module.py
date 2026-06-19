from __future__ import annotations

import math
from typing import Any

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MeanMetric

from src.models.gpt import GPTLikeModel, MaskedLanguageModelingLoss


class GPTLightningModule(LightningModule):
    def __init__(self, config: DictConfig | dict[str, Any]) -> None:
        super().__init__()
        self.config = OmegaConf.create(config)
        self.save_hyperparameters(OmegaConf.to_container(self.config, resolve=True))
        self.model = GPTLikeModel(self.config.model)
        self.loss_fn = MaskedLanguageModelingLoss()
        self.train_epoch_loss = MeanMetric()
        self.val_epoch_loss = MeanMetric()

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        return self.model(input_ids, segment_ids, **model_kwargs)

    def _step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["segment_ids"])
        loss, valid_tokens = self.loss_fn.forward_with_count(
            logits,
            batch["input_ids"],
            batch["segment_ids"],
        )
        metric = self.train_epoch_loss if prefix == "train" else self.val_epoch_loss
        metric.update(loss.detach(), weight=valid_tokens.detach())
        if prefix == "train":
            self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def on_train_epoch_end(self) -> None:
        loss = self.train_epoch_loss.compute()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_perplexity", torch.exp(loss.clamp(max=20)), prog_bar=True)
        self.train_epoch_loss.reset()

    def on_validation_epoch_end(self) -> None:
        loss = self.val_epoch_loss.compute()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", torch.exp(loss.clamp(max=20)), prog_bar=True)
        self.val_epoch_loss.reset()

    def on_after_backward(self) -> None:
        total_norm_squared = torch.zeros((), device=self.device)
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            grad_norm = parameter.grad.detach().norm(2)
            total_norm_squared += grad_norm.pow(2)
            if self.global_step % int(self.config.training.log_every_n_steps) == 0:
                safe_name = name.replace(".", "/")
                self.log(f"grad_norm/{safe_name}", grad_norm, on_step=True, on_epoch=False)
        self.log("grad_norm/global", total_norm_squared.sqrt(), on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict[str, Any]:
        train_config = self.config.training
        optimizer = AdamW(
            self.parameters(),
            lr=float(train_config.learning_rate),
            weight_decay=float(train_config.weight_decay),
        )

        warmup_steps = int(train_config.warmup_steps)
        min_lr_ratio = float(train_config.min_lr_ratio)
        total_steps = max(int(self.trainer.estimated_stepping_batches), warmup_steps + 1)

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, 1e-8)
            progress = max(step - warmup_steps, 0)
            decay_steps = max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(min(progress / decay_steps, 1.0) * math.pi))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
