
"""
SRCNN Training Engine.

Features:
- YAML configuration
- Reproducibility
- CPU/GPU device selection
- Training and validation
- Adam optimizer
- MSE loss
- Best model checkpointing
"""

import random
from pathlib import Path

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader, random_split

from src.models.srcnn import SRCNN
from src.data.srcnn_dataset import SRCNNDataset
from src.losses.srcnn_loss import SRCNNLoss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "srcnn.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config):

    use_cuda = config["device"]["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()
    total_loss = 0.0

    for lr_images, hr_images in loader:

        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        optimizer.zero_grad()

        predictions = model(lr_images)

        loss = criterion(predictions, hr_images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.inference_mode():

        for lr_images, hr_images in loader:

            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)

            predictions = model(lr_images)

            loss = criterion(predictions, hr_images)

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    config = load_config()

    set_seed(config["experiment"]["seed"])

    device = get_device(config)

    print("=" * 60)
    print("SRCNN TRAINING")
    print("=" * 60)

    print("Device:", device)

    # Dataset
    hr_dir = PROJECT_ROOT / config["data"]["hr_directory"]
    lr_dir = PROJECT_ROOT / config["data"]["lr_directory"]

    dataset = SRCNNDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scale_factor=config["data"]["scale_factor"],
        input_crop=config["data"]["input_crop"],
    )

    print("Total dataset images:", len(dataset))

    # Temporary split for pipeline testing only.
    # Do not use a one-image split for scientific evaluation.
    if len(dataset) < 2:
        raise ValueError(
            "At least two image pairs are required for a training/"
            "validation smoke test. Expand the dataset first."
        )

    train_size = max(1, int(0.8 * len(dataset)))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(
            config["experiment"]["seed"]
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Model
    model = SRCNN().to(device)

    # Loss
    criterion = SRCNNLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Checkpoints
    checkpoint_dir = (
        PROJECT_ROOT / config["checkpoint"]["directory"]
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_validation_loss = float("inf")

    # Training loop
    for epoch in range(config["training"]["epochs"]):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/"
            f"{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.6f} "
            f"Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint_path = checkpoint_dir / "best_srcnn.pth"

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": validation_loss,
                    "config": config,
                },
                checkpoint_path,
            )

            print("Best checkpoint saved:", checkpoint_path)

    print("\nSRCNN training completed.")


if __name__ == "__main__":
    main()
