import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import argparse
import logging
import sys
import ipdb

from diffusion.ddpm import DDPM
from diffusion.ddim import DDIM
from models import UNet
from utils import set_global_random_seed


def get_default_configs():
    """Get default configuration dictionaries"""

    data_config = {
        "name": "mnist",
        "root": "/projects/p32013/WJK/learning_diffusion/data/",
        "mean": [0.1307],
        "std": [0.3081],
        # get mean and std from https://datascience.stackexchange.com/questions/46228/how-mean-and-deviation-come-out-with-mnist-dataset
        
        # 方式1：指定具体数量
        # "subset_size": 100,  # 使用1000张图片
        
        # 或者 方式2：指定比例
        # "subset_ratio": 0.1,  # 使用10%的数据
        
        # 可选：是否随机采样
        "subset_random": True,  # 随机采样，False则按顺序取
    }

    model_config = {
        "type": "ddim",  # 'ddpm' or 'ddim'
        "n_steps": 1000,
        # DDIM specific configs
        "ddim_sampling_steps": 1000,
        "ddim_discretize": "uniform",  # "uniform" or "quad"
        "ddim_eta": 0.0,
    }

    training_config = {
        "batch_size": 512,
        "num_workers": 5,
        "learning_rate": 2e-3,
        "n_epochs": 50,
        "save_interval": 1,  # epoch
        "log_interval": 100,  # step
        "save_path": "/projects/p32013/WJK/learning_diffusion/results/",
        "seed": 42,
    }

    unet_config = {
        "channels": [10, 20,40 ,80],
        "pe_dim": 10,
        "residual": True,
    }

    return {
        "data": data_config,
        "model": model_config,
        "training": training_config,
        "unet": unet_config,
    }


def save_config(config, path):
    """Save configuration to JSON file"""
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def load_config(path):
    """Load configuration from JSON file"""
    with open(path, "r") as f:
        return json.load(f)


def get_dataset(data_config):
    """
    Initialize dataset based on configuration
    
    Args:
        data_config: dict containing:
            - name: dataset name
            - root: data root directory
            - mean: normalization mean
            - std: normalization std
            - subset_size: (optional) number of samples to use
            - subset_ratio: (optional) ratio of data to use (0-1)
            - subset_random: (optional) whether to randomly sample (default: True)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_config["mean"], data_config["std"]),
    ])

    if data_config["name"].lower() == "mnist":
        full_dataset = datasets.MNIST(
            data_config["root"], 
            train=True, 
            download=True, 
            transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {data_config['name']} not implemented")

    # 处理数据子集
    if "subset_size" in data_config or "subset_ratio" in data_config:
        total_size = len(full_dataset)
        
        # 确定子集大小
        if "subset_size" in data_config:
            subset_size = min(data_config["subset_size"], total_size)
        else:
            subset_size = int(total_size * data_config["subset_ratio"])
            
        # 是否随机采样
        random_subset = data_config.get("subset_random", True)
        
        if random_subset:
            indices = torch.randperm(total_size)[:subset_size]
        else:
            indices = torch.arange(subset_size)
            
        dataset = torch.utils.data.Subset(full_dataset, indices)
        
        print(f"Using {subset_size} samples out of {total_size} "
              f"({'random' if random_subset else 'sequential'} sampling)")
    else:
        dataset = full_dataset
        print(f"Using full dataset with {len(dataset)} samples")

    return dataset


def get_diffusion_model(model_config, eps_model, device):
    """Initialize diffusion model based on configuration"""
    if model_config["type"].lower() == "ddpm":
        return DDPM(eps_model=eps_model, n_step=model_config["n_steps"], device=device)
    elif model_config["type"].lower() == "ddim":
        return DDIM(
            eps_model=eps_model,
            n_step=model_config["n_steps"],
            ddim_sampling_steps=model_config["ddim_sampling_steps"],
            ddim_discretize=model_config["ddim_discretize"],
            ddim_eta=model_config["ddim_eta"],
            device=device,
        )
    else:
        raise NotImplementedError(f"Model type {model_config['type']} not implemented")


def setup_logger(save_dir):
    """设置logger，同时输出到文件和控制台"""
    logger = logging.getLogger("diffusion_training")
    logger.setLevel(logging.DEBUG)

    # 创建格式器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件处理器 - DEBUG及以上级别
    fh = logging.FileHandler(save_dir / "debug.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 控制台处理器 - INFO及以上级别
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train_and_sample(config):
    # 设置路径
    tic = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['model']['type']}_{config['data']['name']}_{tic}"
    save_dir = Path(config["training"]["save_path"]) / experiment_name
    save_image_folder = save_dir / "samples"
    ckpt_folder = save_dir / "checkpoint"
    config_path = save_dir / "config.json"
    ckpt_path = ckpt_folder / "final_model.pth"

    save_image_folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)

    # 设置logger
    logger = setup_logger(save_dir)

    # 保存配置
    save_config(config, config_path)
    logger.debug(f"Configuration saved to {config_path}")

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 数据集设置
    dataset = get_dataset(config["data"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    logger.debug(
        f"Dataset size: {len(dataset)}, Batch size: {config['training']['batch_size']}"
    )

    # 模型设置
    eps_model = UNet(
        n_steps=config["model"]["n_steps"],
        image_shape=tuple(dataset[0][0].shape),
        **config["unet"],
    ).to(device)

    diffusion_model = get_diffusion_model(config["model"], eps_model, device)
    logger.info(f"Initialized {config['model']['type'].upper()} model")
    logger.debug(f"Model parameters: {sum(p.numel() for p in eps_model.parameters())}")

    # 优化器设置
    optimizer = optim.Adam(
        eps_model.parameters(), lr=config["training"]["learning_rate"]
    )
    logger.debug(
        f"Optimizer: Adam with learning rate {config['training']['learning_rate']}"
    )

    # 训练循环
    n_epochs = config["training"]["n_epochs"]
    save_interval = config["training"]["save_interval"]
    log_interval = config["training"]["log_interval"]

    logger.info("Starting training...")
    for epoch in range(n_epochs):
        total_loss = 0
        # # ipdb.set_trace()  # 这里会触发断点
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            loss = diffusion_model.loss(data)
            loss.backward()
            optimizer.step()
            # # ipdb.set_trace()  # 这里会触发断点

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            logger.info(f"Generating samples for epoch {epoch}...")
            # ipdb.set_trace()  # 这里会触发断点
            with torch.no_grad():
                samples = diffusion_model.sample_backforward(
                    shape=(16, *tuple(dataset[0][0].shape))
                )
                # ipdb.set_trace()  # 这里会触发断点

                plt.figure(figsize=(10, 10))
                for i in range(16):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
                    plt.axis("off")
                sample_path = save_image_folder / f"samples_epoch_{epoch}.png"
                plt.savefig(sample_path)
                plt.close()
                logger.debug(f"Saved samples to {sample_path}")

    # 保存最终模型
    logger.info("Saving final model...")
    torch.save(
        {
            "model_state_dict": eps_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        ckpt_path,
    )
    logger.info(f"Model saved to {ckpt_path}")
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_configs()
    set_global_random_seed(config['training']['seed'])
    train_and_sample(config)
