# ruff: noqa: F401
from typing import Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from SDE import BaseSDE, VPSDE, VESDE, SubVPSDE
from ScoreModel import ScoreModel


class SdeTrainer:
    def __init__(
        self,
        score_model: ScoreModel,
        optimizer: Optimizer,
    ):
        """SDE训练器

        Args:
            score_model: ScoreModel实例
            optimizer: 优化器
        """
        self.score_model = score_model
        self.optimizer = optimizer

    def train_step(self, batch: torch.Tensor) -> dict:
        """单步训练

        Args:
            batch: 输入数据批次

        Returns:
            包含损失值的字典
        """
        self.optimizer.zero_grad()
        loss = self.score_model.train_step(batch)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        log_interval: int = 100,
    ):
        """训练循环"""
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.score_model.device)
                metrics = self.train_step(batch)

                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch {epoch}, Step {batch_idx}, Loss: {metrics['loss']:.4f}"
                    )
