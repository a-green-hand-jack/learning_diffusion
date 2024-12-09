from dataclasses import dataclass
from email.headerregistry import DateHeader
from typing import Tuple

import torchvision.transforms as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


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
        return DateHeader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
