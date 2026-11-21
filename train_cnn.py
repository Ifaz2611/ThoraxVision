"""Optional transfer-learning experiment: fine-tune a small CNN on the CT data.

This is intentionally separate from the main deployment path so the Docker
image and the default app stay small. Install the extra dependencies with::

    pip install -r requirements-cnn.txt

Then train and compare against the RandomForest baseline::

    python train_cnn.py --dataset dataset/dataset --epochs 8
    python evaluate.py --model model_cnn.pt --split test
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from class_info import canonical_class_name
from dataset_utils import (
    IMAGE_EXTENSIONS,
    compute_metrics,
    format_metrics,
    resolve_dataset_splits,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision import models, transforms
except ImportError as exc:  # pragma: no cover - exercised without torch
    raise SystemExit(
        "train_cnn.py requires torch and torchvision. Install them with:\n"
        "  pip install -r requirements-cnn.txt"
    ) from exc

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / "model_cnn.pt"
DEFAULT_METRICS = BASE_DIR / "reports" / "cnn_metrics.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a MobileNetV2/ResNet18 on the CT dataset."
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=3,
        help="Epochs to train only the new classifier head.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--arch",
        choices=["mobilenet_v2", "resnet18"],
        default="mobilenet_v2",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download ImageNet weights (not transfer learning).",
    )
    return parser.parse_args()


class CTDataset(Dataset):
    """ImageFolder-like dataset with canonical class-name mapping."""

    def __init__(self, split_dir: Path, classes: list[str], transform):
        self.classes = list(classes)
        class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_dir in sorted(
            path for path in Path(split_dir).iterdir() if path.is_dir()
        ):
            label = canonical_class_name(class_dir.name)
            if label not in class_to_index:
                continue
            for image_path in sorted(class_dir.iterdir()):
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    self.samples.append((image_path, class_to_index[label]))

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label


def build_model(arch: str, num_classes: int, pretrained: bool):
    if pretrained:
        weights = (
            models.MobileNet_V2_Weights.DEFAULT
            if arch == "mobilenet_v2"
            else models.ResNet18_Weights.DEFAULT
        )
    else:
        weights = None

    if arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=weights)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        model = models.resnet18(weights=weights)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def unfreeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def evaluate(model, loader, device, classes):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(predictions.tolist())
            y_true.extend(labels.numpy().tolist())
    return compute_metrics(y_true, y_pred, classes)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    try:
        splits = resolve_dataset_splits(args.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    train_dir = splits["train"]
    classes = sorted(
        {
            canonical_class_name(path.name)
            for path in train_dir.iterdir()
            if path.is_dir()
        }
    )
    print(f"Train split: {train_dir}")
    print("Classes:", classes)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = CTDataset(train_dir, classes, train_transform)
    valid_dir = splits.get("valid")
    if valid_dir is not None:
        valid_dataset = CTDataset(valid_dir, classes, eval_transform)
    else:
        # Fall back to a held-out slice of the training data.
        validation_size = max(1, int(0.15 * len(train_dataset)))
        train_size = len(train_dataset) - validation_size
        train_dataset, valid_dataset = random_split(
            train_dataset,
            [train_size, validation_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(
            f"No valid/ split found; holding out {validation_size} training "
            "images for validation."
        )

    test_dataset = None
    test_dir = splits.get("test")
    if test_dir is not None:
        test_dataset = CTDataset(test_dir, classes, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        if test_dataset is not None
        else None
    )

    model = build_model(
        args.arch, len(classes), pretrained=not args.no_pretrained
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    def make_optimizer(parameters, lr):
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    optimizer = make_optimizer(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    best_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            unfreeze(model)
            optimizer = make_optimizer(model.parameters(), args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - epoch + 1)
            )
            print("Unfroze the backbone for fine-tuning.")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_accuracy = correct / total
        valid_metrics = evaluate(model, valid_loader, device, classes)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_metrics["accuracy"],
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} "
            f"valid_acc={valid_metrics['accuracy']:.4f} "
            f"valid_f1={valid_metrics['macro_f1']:.4f}"
        )

        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = {"valid": evaluate(model, valid_loader, device, classes)}
    if test_loader is not None:
        final_metrics["test"] = evaluate(model, test_loader, device, classes)
        print(format_metrics(final_metrics["test"], title="Test metrics"))
    print(format_metrics(final_metrics["valid"], title="Validation metrics"))

    checkpoint = {
        "arch": args.arch,
        "state_dict": model.state_dict(),
        "classes": classes,
        "image_size": args.image_size,
        "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": final_metrics,
        "history": history,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved CNN checkpoint to {output}")

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "arch": args.arch,
                "classes": classes,
                "image_size": args.image_size,
                "trained_at": checkpoint["trained_at"],
                "metrics": final_metrics,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
