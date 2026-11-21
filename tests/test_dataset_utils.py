from pathlib import Path

import numpy as np
from PIL import Image

from dataset_utils import (
    compute_metrics,
    discover_classes,
    iter_class_images,
    load_split,
    resolve_dataset_splits,
)


def make_split(root: Path, split: str, class_names=("alpha", "beta")):
    split_dir = root / "Data" / split
    for index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        Image.new("L", (16, 16), color=40 + index * 60).save(
            class_dir / f"{class_name}_1.png"
        )
    return split_dir


def test_resolve_dataset_splits_from_root(tmp_path):
    train = make_split(tmp_path, "train")
    valid = make_split(tmp_path, "valid")
    test = make_split(tmp_path, "test")

    splits = resolve_dataset_splits(tmp_path)
    assert splits["train"] == train
    assert splits["valid"] == valid
    assert splits["test"] == test


def test_resolve_dataset_splits_from_train_dir(tmp_path):
    train = make_split(tmp_path, "train")
    make_split(tmp_path, "valid")
    make_split(tmp_path, "test")

    splits = resolve_dataset_splits(train)
    assert splits["train"] == train
    assert splits["valid"] is not None
    assert splits["test"] is not None


def test_resolve_dataset_splits_from_nested_repo_layout(tmp_path):
    nested = tmp_path / "dataset" / "dataset"
    train = make_split(nested, "train")
    make_split(nested, "valid")
    make_split(nested, "test")

    splits = resolve_dataset_splits(tmp_path)
    assert splits["train"] == train
    assert splits["valid"] is not None
    assert splits["test"] is not None


def test_load_split_and_class_discovery(tmp_path):
    train = make_split(tmp_path, "train")
    classes = discover_classes(train)
    assert classes == ["alpha", "beta"]

    features, labels, loaded_classes, skipped = load_split(
        train, image_size=8, classes=classes
    )
    assert features.shape == (2, 64)
    assert labels.tolist() == [0, 1]
    assert loaded_classes == classes
    assert skipped == 0


def test_iter_class_images_uses_canonical_names(tmp_path):
    split = tmp_path / "split"
    class_dir = split / "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib"
    class_dir.mkdir(parents=True)
    Image.new("L", (8, 8)).save(class_dir / "image.png")

    pairs = list(iter_class_images(split))
    assert len(pairs) == 1
    assert pairs[0][1] == "adenocarcinoma"


def test_compute_metrics():
    metrics = compute_metrics([0, 1, 1, 0], [0, 1, 0, 0], ["a", "b"])
    assert metrics["accuracy"] == 0.75
    assert metrics["n_samples"] == 4
    assert metrics["per_class"]["a"]["support"] == 2
    assert metrics["per_class"]["b"]["recall"] == 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert np.isclose(metrics["macro_f1"], 0.7333333333333334)
