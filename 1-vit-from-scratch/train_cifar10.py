from pathlib import Path
import argparse
import json

from datasets import load_dataset
import torch

import data
import models
import training


N_CLASSES = 10
IMAGE_SIZE = 32
N_CHANNELS = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Vision Transformer on CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("output_dir", type=Path, help="folder for checkpoint.pth, history.csv and config.json")

    model_args = parser.add_argument_group("model parameters")
    model_args.add_argument("--patch-size", type=int, default=4, help="side of the square patches, must divide 32")
    model_args.add_argument("--embedding-dim", type=int, default=192, help="token size, must be divisible by --n-heads")
    model_args.add_argument("--n-layers", type=int, default=4, help="number of transformer layers")
    model_args.add_argument("--mlp-dim", type=int, default=768, help="hidden size of the transformer MLP")
    model_args.add_argument("--n-heads", type=int, default=3, help="number of attention heads")
    model_args.add_argument("--dropout", type=float, default=0, help="dropout probability")
    model_args.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True, help="use the fused QKV attention")

    augmentation_args = parser.add_argument_group("data augmentation parameters")
    augmentation_args.add_argument("--flip-p", type=float, default=0, help="probability of a random horizontal flip")
    augmentation_args.add_argument("--crop-padding", type=int, default=0, help="padding in pixels added to each side before the random crop")

    training_args = parser.add_argument_group("training parameters")
    training_args.add_argument("--n-epochs", type=int, default=10, help="number of training epochs")
    training_args.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    training_args.add_argument("--train-batch-size", type=int, default=64, help="batch size for training")
    training_args.add_argument("--test-batch-size", type=int, default=256, help="batch size for validation and test")
    training_args.add_argument("--val-ratio", type=float, default=0.1, help="fraction of the train split held out for validation")
    training_args.add_argument("--log-steps", type=int, default=100, help="steps between train loss logs")
    training_args.add_argument("--seed", type=int, default=0, help="seed for the model init, shuffling and validation split")

    args = parser.parse_args()
    if (args.output_dir / "history.csv").exists():
        parser.error(f"{args.output_dir} already contains a run")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    params = {
        "n_classes": N_CLASSES,
        "image_size": IMAGE_SIZE,
        "n_channels": N_CHANNELS,
        "patch_size": args.patch_size,
        "embedding_dim": args.embedding_dim,
        "n_layers": args.n_layers,
        "mlp_dim": args.mlp_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "fused": args.fused
    }
    model = models.ViTForClassification(**params)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params}")

    ds = load_dataset("uoft-cs/cifar10")
    train_loader, val_loader, test_loader = data.get_cifar10_loaders(
        ds,
        flip_p=args.flip_p,
        crop_padding=args.crop_padding,
        val_ratio=args.val_ratio,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        seed=args.seed
    )

    history = training.train_classif(
        model, train_loader, val_loader, device,
        n_epochs=args.n_epochs,
        log_steps=args.log_steps,
        lr=args.lr
    )

    test_loss, test_acc = training.eval_classif(model, test_loader, device)
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test accuracy: {test_acc*100:.2f}")

    print(f"Saving to {args.output_dir}...")
    torch.save(model.state_dict(), args.output_dir / "checkpoint.pth")
    history.to_csv(args.output_dir / "history.csv", index=False)
    config = {
        **params,
        "n_params": n_params,
        "flip_p": args.flip_p,
        "crop_padding": args.crop_padding,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "test_loss": test_loss,
        "test_acc": test_acc
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
