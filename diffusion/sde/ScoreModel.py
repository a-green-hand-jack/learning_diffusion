from copy import deepcopy
import pytorch_lightning as pl
import torch
import torch.nn as nn
from SDE import VESDE, VPSDE, BaseSDE, SubVPSDE
from torch.optim import Adam


class ScoreModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        sde: BaseSDE | VPSDE | VESDE | SubVPSDE,
    ):
        """Score模型的核心实现

        Args:
            model: 基础模型（如UNet）
            sde: SDE实例
        """
        super().__init__()
        self.model: nn.Module = model
        self.sde: BaseSDE | VPSDE | VESDE | SubVPSDE = sde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算损失

        Args:
            x: 输入数据 [batch_size, *data_shape]

        Returns:
            loss: 标量损失值
        """
        # 采样时间点
        t = torch.rand(x.shape[0], device=x.device)

        # 生成噪声
        noise = torch.randn_like(x)

        # 获取扰动数据
        mean, std = self.sde.marginal_prob(x, t)
        perturbed_data = mean + std[:, None, None, None] * noise

        # 获取模型预测的score
        pred_score = self.model(perturbed_data, t)

        # 获取目标score
        target_score = self.sde.score(x, perturbed_data, t)

        # 获取损失权重
        lambda_t = self.sde.loss_weight(t).view(-1, 1, 1, 1)

        # 计算加权MSE损失
        loss = torch.mean(
            lambda_t * torch.sum((pred_score - target_score) ** 2, dim=(1, 2, 3))
        )

        return loss

    def get_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """获取score预测值

        Args:
            x: 输入数据
            t: 时间点

        Returns:
            预测的score值
        """
        return self.model(x, t)


class LightningScoreModel(pl.LightningModule):
    def __init__(
        self,
        score_model: ScoreModel,
        learning_rate: float = 1e-4,
        ema_decay: float = 0.999,
    ):
        """Lightning包装器，处理训练和采样逻辑

        Args:
            score_model: ScoreModel实例
            learning_rate: 学习率
            ema_decay: EMA模型的衰减率
        """
        super().__init__()
        self.save_hyperparameters(ignore=["score_model"])

        self.score_model = score_model
        self.learning_rate = learning_rate

        # 创建EMA模型
        self.ema_score_model: ScoreModel = deepcopy(score_model)
        for param in self.ema_score_model.parameters():
            param.requires_grad_(False)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss = self.score_model(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.update_ema()
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        loss = self.score_model(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """配置优化器"""
        return Adam(self.score_model.parameters(), lr=self.learning_rate)

    @torch.no_grad()
    def update_ema(self):
        """更新EMA模型参数"""
        for ema_param, model_param in zip(
            self.ema_score_model.parameters(), self.score_model.parameters()
        ):
            ema_param.data.mul_(self.hparams.ema_decay)
            ema_param.data.add_((1 - self.hparams.ema_decay) * model_param.data)

    @torch.no_grad()
    def sample(self, shape, device, num_steps=1000):
        """采样方法

        Args:
            shape: 采样形状
            device: 设备
            num_steps: 采样步数

        Returns:
            采样得到的图像
        """
        # 使用EMA模型进行采样
        sde = self.score_model.sde

        # 从标准正态分布采样
        x = torch.randn(shape, device=device)

        # 逆向扩散过程
        timesteps = torch.linspace(1.0, 0.0, num_steps, device=device)
        for t in timesteps:
            # 扩展时间维度以匹配批次大小
            t_batch = t.repeat(shape[0])

            # 获取score
            score = self.ema_score_model.get_score(x, t_batch)

            # 更新x（这里需要根据具体的SDE类型实现相应的采样步骤）
            x = sde.reverse_step(x, score, t)

        return x

    def on_save_checkpoint(self, checkpoint):
        """保存检查点"""
        checkpoint["ema_state_dict"] = self.ema_score_model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """加载检查点"""
        self.ema_score_model.load_state_dict(checkpoint["ema_state_dict"])
