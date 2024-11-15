from dataclasses import dataclass
from typing import Optional, Tuple

from lightning import LightningDataModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from diffusion import (
    ConsistencySamplingAndEditing,
    ConsistencyTraining,
    LitConsistencyModel,
    LitConsistencyModelConfig,
)
from model import AttUNet, AttUNetConfig, UNet, UNetConfig


@dataclass
class ImageDataModuleConfig:
    data_dir: str = "butterflies256"
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


class ImageDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str = None) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        self.dataset = ImageFolder(self.config.data_dir, transform=transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )


@dataclass
class TrainingConfig:
    image_dm_config: ImageDataModuleConfig
    UNet_config: UNetConfig
    consistency_training: ConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_cm_config: LitConsistencyModelConfig
    trainer: Trainer
    seed: int = 42
    model_ckpt_path: str = "./results/cm"
    resume_ckpt_path: Optional[str] = None


def run_training(config: TrainingConfig) -> None:
    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = ImageDataModule(config.image_dm_config)

    # Create student and teacher models and EMA student model
    student_model = UNet(config.UNet_config)
    teacher_model = UNet(config.UNet_config)
    teacher_model.load_state_dict(student_model.state_dict())
    ema_student_model = UNet(config.UNet_config)
    ema_student_model.load_state_dict(student_model.state_dict())

    # Create lightning module
    lit_cm = LitConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        student_model,
        teacher_model,
        ema_student_model,
        config.lit_cm_config,
    )

    # Run training
    config.trainer.fit(lit_cm, dm, ckpt_path=config.resume_ckpt_path)

    # Save model
    lit_cm.ema_student_model.save_pretrained(config.model_ckpt_path)


if __name__ == "__main__":
    training_config = TrainingConfig(
        image_dm_config=ImageDataModuleConfig(
            "/projects/p32013/WJK/DIFFUSION/learning_diffusion/data/butterflies256/",
            image_size=(256, 256),
            num_workers=4,
        ),
        UNet_config=UNetConfig(image_shape=(3, 256, 256)),
        consistency_training=ConsistencyTraining(final_timesteps=17),
        consistency_sampling=ConsistencySamplingAndEditing(),
        lit_cm_config=LitConsistencyModelConfig(
            sample_every_n_steps=1000, lr_scheduler_iters=1000
        ),
        trainer=Trainer(
            max_steps=10_000,
            precision="16-mixed",
            log_every_n_steps=10,
            logger=TensorBoardLogger(".", name="logs", version="cm"),
            callbacks=[LearningRateMonitor(logging_interval="step")],
        ),
    )
    run_training(training_config)
