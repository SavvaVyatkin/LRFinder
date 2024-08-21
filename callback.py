from typing import Any, Literal, Mapping

import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.optimizer import _validate_optimizers_attached
from lightning.pytorch.utilities.types import LRSchedulerConfig
from torch.optim.lr_scheduler import ExponentialLR, LinearLR


# TODO: `max_steps` parameter to LRFinder instead of using pl.Trainer's max_epoch.
class LRFinder(Callback):
    def __init__(
        self,
        min_lr: float,
        max_lr: float,
        lr_type: Literal["exponential", "linear"] = "exponential",
        beta: float = 0.98,
    ):
        self._val_loss = []
        self._train_loss = []
        self._train_lrs = []
        self._val_lrs = []
        self.lr_type = lr_type
        self.min_lr = min_lr
        self.max_lr = max_lr

        # for loss smoothing
        self.avg_loss = 0.0
        self.beta = beta

    @property
    def val_loss(self):
        return np.asarray(self._val_loss)

    @property
    def train_loss(self):
        return np.asarray(self._train_loss)

    @property
    def val_lrs(self):
        return np.asarray(self._val_lrs)

    @property
    def train_lrs(self):
        return np.asarray(self._train_lrs)

    def _exchange_scheduler(self, trainer) -> None:
        optimizers = trainer.strategy.optimizers

        if len(optimizers) != 1:
            raise ValueError(
                f"`model.configure_optimizers()` returned {len(optimizers)}, but"
                " learning rate finder only works with single optimizer"
            )

        optimizer = optimizers[0]

        if self.lr_type == "exponential":

            for group in optimizer.param_groups:
                group["lr"] = self.min_lr
                group["initial_lr"] = self.min_lr

            gamma = np.exp(
                np.log(self.max_lr / self.min_lr)
                / int(trainer.estimated_stepping_batches)
            )
            scheduler = ExponentialLR(optimizer, gamma)
        elif self.lr_type == "linear":
            for group in optimizer.param_groups:
                group["lr"] = self.max_lr
                group["initial_lr"] = self.max_lr

            scheduler = LinearLR(
                optimizer,
                self.min_lr / self.max_lr,
                1.0,
                total_iters=trainer.estimated_stepping_batches,
            )
        else:
            raise NotImplementedError(f"Unknown lr_type {self.lr_type}")

        trainer.strategy.optimizers = [optimizer]
        trainer.strategy.lr_scheduler_configs = [
            LRSchedulerConfig(scheduler, interval="step")
        ]
        _validate_optimizers_attached(trainer.optimizers, trainer.lr_scheduler_configs)

    def _get_lr(self, trainer) -> float:
        return trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]

    def on_fit_start(self, trainer, pl_module):
        self._exchange_scheduler(trainer)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        # Account for gradient accumulation
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        self._train_lrs.append(self._get_lr(trainer))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
    ) -> None:
        # Account for gradient accumulation
        if (trainer.fit_loop.batch_idx + 1) % trainer.accumulate_grad_batches != 0:
            return

        # _AutomaticOptimization.run turns None STEP_OUTPUT into an empty dict
        if not outputs:
            print("not outputs")
            # need to add an element, because we also added one element to lrs in on_train_batch_start
            # so add nan, because they are not considered when computing the suggestion
            self._train_loss.append(float("nan"))
            return

        current_loss = outputs["loss"].item()
        current_step = trainer.global_step

        # Avg loss (loss with momentum) + smoothing
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * current_loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** (current_step + 1))

        self._train_loss.append(smoothed_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._val_lrs.append(self._get_lr(trainer))

        self._val_loss.append(np.mean(pl_module.validation_outputs["loss"]))
