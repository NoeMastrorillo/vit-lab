import functools

from datasets import load_dataset, DatasetDict
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader


def get_cifar10_loaders(dataset, flip_p=0, crop_padding=0, val_ratio=0.1, train_batch_size=64, test_batch_size=256, seed=0):
    imgs = np.stack([np.asarray(img) for img in dataset["train"]["img"]]).astype(np.float32)
    means = np.array(imgs.mean(axis=(0, 1, 2), dtype=np.float64).tolist())
    stds = np.array(imgs.std(axis=(0, 1, 2), dtype=np.float64).tolist())

    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(flip_p),
        v2.RandomCrop(32, crop_padding, padding_mode="reflect"),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(means/255, stds/255)
    ])

    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(means/255, stds/255)
    ])

    def transform_batch(transform, batch):
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    split_ds = dataset["train"].train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split_ds["train"].with_transform(functools.partial(transform_batch, train_transform))
    val_ds = split_ds["test"].with_transform(functools.partial(transform_batch, test_transform))
    test_ds = dataset["test"].with_transform(functools.partial(transform_batch, test_transform))

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


