import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.utils import ImageDataModule
from diffusion.sde.ScoreModel import ScoreModel, LightningScoreModel
from diffusion.sde.SDE import get_sde
from models.unet import UNet

@hydra.main(config_path="configs", config_name="sde_base")
def run_training(cfg: DictConfig) -> None:
    # 设置随机种子
    pl.seed_everything(cfg.seed)

    # 创建数据模块
    data_module = ImageDataModule(**cfg.data)

    # 创建模型
    model = UNet(**cfg.model.params)
    
    # 创建SDE实例
    sde = get_sde(cfg.sde.name)(**cfg.sde.params)
    
    # 创建ScoreModel
    score_model = ScoreModel(
        model=model,
        sde=sde
    )
    
    # 创建LightningScoreModel
    lit_score_model = LightningScoreModel(
        score_model=score_model,
        learning_rate=cfg.training.optimizer.lr,
        ema_decay=cfg.training.ema_decay
    )

    # 创建回调
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            **cfg.training.checkpoint,
            dirpath=f"{HydraConfig.get().runtime.output_dir}/{cfg.training.checkpoint.dirpath}"
        )
    ]

    # 创建logger
    logger = TensorBoardLogger(
        save_dir=f"{HydraConfig.get().runtime.output_dir}/{cfg.logging.save_dir}",
        name=cfg.logging.name
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        val_check_interval=cfg.training.val_check_interval,
        logger=logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1
    )

    # 开始训练
    trainer.fit(lit_score_model, data_module)

if __name__ == "__main__":
    run_training()
