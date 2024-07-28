"""
Main file for forward-forward algorithm (https://arxiv.org/abs/2212.13345).
"""

import os
import logging
import pprint

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

from src.datasets import mnist, cifar
from src.models.forward_forward_block import LabelEmbedder
from src.models.forward_forward_model import FFModel, FFMLP, FFCNN


def calc_accuracy(model, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += model.predict(x).eq(y).sum().item()
    return correct / len(loader.dataset)


def train(cfg, model: FFModel, train_loader: DataLoader, device: str):
    model.train()
    if cfg["train_method"] == "simultaneous":
        model.train_model_simultaneously(train_loader=train_loader, num_epochs=cfg["num_epochs"], device=device)
    elif cfg["train_method"] == "sequential":
        model.train_model_sequentially(train_loader=train_loader, num_epochs=cfg["num_epochs"], device=device)
    else:
        raise ValueError("Unknown train method: {}".format(cfg["train_method"]))



@hydra.main(config_path="conf", config_name="main_ff", version_base=None)
def main(cfg: OmegaConf) -> None:
    seed = cfg["seed"]
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(cfg["gpu"]))
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Dataset
    dataset_name = cfg["dataset"]["name"]
    if dataset_name == "mnist":
        train_dataset, valid_dataset, test_dataset = mnist.get_MNIST_datasets()
    elif dataset_name == "cifar10" or dataset_name == "cifar100":
        train_dataset, valid_dataset, test_dataset = cifar.get_CIFAR_datasets(dataset_name=dataset_name)
    else:
        raise ValueError("Dataset name must be 'mnist', 'cifar10' or 'cifar100'.")

    batch_size = cfg["dataset"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg["num_workers"], pin_memory=True
    )
    _ = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )

    label_embedder = LabelEmbedder(train_loader=train_loader, num_classes=cfg["dataset"]["num_classes"], method=cfg["label_embedder"]["method"], device=device)

    # Model
    model_name = cfg["model"]["name"]
    label_embedder_method = cfg["label_embedder"]["method"]
    num_channels = cfg["dataset"]["num_channels"]
    if model_name == "mlp":
        if label_embedder_method.endswith("2channel"):
            raise ValueError("{} cannot be specified for MLP.".format(label_embedder_method))
        model = FFMLP(
            lr=cfg["optimizer"]["learning_rate"],
            opt_name=cfg["optimizer"]["name"],
            threshold=cfg["threshold"],
            num_classes=cfg["dataset"]["num_classes"],
            dims=[cfg["dataset"]["size"] * cfg["dataset"]["size"] * num_channels, 1000, 1000, 1000, 1000, cfg["dataset"]["num_classes"]],
            label_embedder=label_embedder,
            device=device,
        )
    elif model_name == "cnn":
        if label_embedder_method.endswith("2channel"):
            num_channels = 2 * num_channels
        model = FFCNN(
            lr=cfg["optimizer"]["learning_rate"],
            opt_name=cfg["optimizer"]["name"],
            threshold=cfg["threshold"],
            num_classes=cfg["dataset"]["num_classes"],
            image_width=cfg["dataset"]["size"],
            label_embedder=label_embedder,
            num_channels=num_channels,
            device=device,
        )
    else:
        raise ValueError("Model name must be 'mlp' or 'cnn'.")


    # Set the path to save the trained model
    model_save_path = os.path.join(os.getcwd(), "save/forward_forward_model/{}/".format(dataset_name))
    model_save_name = "{}_{}_lr_{}_bsz_{}_{}".format(
        cfg["model"]["name"],
        cfg["label_embedder"]["method"],
        cfg["optimizer"]["learning_rate"],
        cfg["dataset"]["batch_size"],
        cfg["train_method"]
    )
    model_save_folder = os.path.join(model_save_path, model_save_name, cfg["id"], "trial_{}".format(cfg["trial"]))
    if not os.path.isdir(model_save_folder):
        os.makedirs(model_save_folder)

    # Logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join(model_save_folder, "log.txt"), mode="w"))
    logger.info("Model information: {}".format(model))
    with open(os.path.join(model_save_folder, "hyperparameter.txt"), mode="w") as f:
        pprint.pprint(cfg, f)

    # Training
    model.train()
    train(cfg=cfg, model=model, train_loader=train_loader, device=device)

    train_acc = 100.0 * calc_accuracy(model=model, loader=train_loader, device=device)
    test_acc = 100.0 * calc_accuracy(model=model, loader=test_loader, device=device)

    logger.info({"train_acc": train_acc, "test_acc": test_acc})


    # Save the last model
    torch.save(model.state_dict(), os.path.join(model_save_folder, "last.pth"))

    pprint.pprint(cfg)
    print("Output saved to {}".format(model_save_folder))


if __name__ == "__main__":
    main()
