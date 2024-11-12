import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import json
import argparse

from diffusion.ddpm import DDPM
from diffusion.ddim import DDIM
from model import UNet


def get_default_configs():
    """Get default configuration dictionaries"""
    data_config = {
        "name": "mnist",
        "root": "/projects/p32013/WJK/learning_diffusion/data/",
        "mean": [0.5],
        "std": [0.5],
    }

    model_config = {
        "type": "ddim",  # 'ddpm' or 'ddim'
        "n_steps": 1000,
        # DDIM specific configs
        "ddim_sampling_steps": 50,
        "ddim_discretize": "quad",  # "uniform" or "quad"
        "ddim_eta": 0.0,
    }

    training_config = {
        "batch_size": 512,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "n_epochs": 50,
        "save_interval": 1,
        "log_interval": 100,
        'save_path':"/projects/p32013/WJK/learning_diffusion/results/"
    }

    unet_config = {
        "channels": [10, 20, 40, 80],
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
    """Initialize dataset based on configuration"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(data_config["mean"], data_config["std"]),
        ]
    )

    if data_config["name"].lower() == "mnist":
        dataset = datasets.MNIST(
            data_config["root"], train=True, download=True, transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset {data_config['name']} not implemented")

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


def train_and_sample(config):
    # Set up paths
    tic = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['model']['type']}_{config['data']['name']}_{tic}"
    save_dir = Path(config["training"]['save_path']) / experiment_name
    save_image_folder = save_dir / "samples"
    ckpt_folder = save_dir / "checkpoint"
    config_path = save_dir / "config.json"
    ckpt_path = ckpt_folder / "final_model.pth"

    save_image_folder.mkdir(parents=True, exist_ok=True)
    ckpt_folder.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, config_path)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset setup
    dataset = get_dataset(config["data"])
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    # Model setup
    eps_model = UNet(
        n_steps=config["model"]["n_steps"],
        image_shape=tuple(dataset[0][0].shape),
        **config["unet"],
    ).to(device)

    diffusion_model = get_diffusion_model(config["model"], eps_model, device)

    # Optimizer setup
    optimizer = optim.Adam(
        eps_model.parameters(), lr=config["training"]["learning_rate"]
    )

    # Training loop
    n_epochs = config["training"]["n_epochs"]
    save_interval = config["training"]["save_interval"]
    log_interval = config["training"]["log_interval"]

    for epoch in range(n_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            loss = diffusion_model.loss(data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            print("Generating samples...")
            with torch.no_grad():
                samples = diffusion_model.sample_backforward(
                    shape=(16, *dataset[0][0].shape)
                )

                plt.figure(figsize=(10, 10))
                for i in range(16):
                    plt.subplot(4, 4, i + 1)
                    plt.imshow(samples[i, 0].cpu().numpy(), cmap="gray")
                    plt.axis("off")
                plt.savefig(save_image_folder / f"samples_epoch_{epoch}.png")
                plt.close()

    # Save final model
    torch.save(
        {
            "model_state_dict": eps_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        ckpt_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    args = parser.parse_args()

    if args.config:
        # Load configuration from JSON file
        config = load_config(args.config)
    else:
        # Use default configuration
        config = get_default_configs()

    # Start training
    train_and_sample(config)
