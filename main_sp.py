"""
Main file for signal propagation algorithm (https://arxiv.org/abs/2204.01723), which is one of the layer-wise training methods.
"""

import os
import logging
import pprint

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from src.datasets import mnist, cifar
from src.util import calc_accuracy, AverageMeter
from src.models.signal_propagation_model import (
    SPForwardResult,
    SPLayerWiseCNN,
    SPLayerWiseResNet,
    SPLayerWiseVGG,
)


@hydra.main(config_path="conf", config_name="main_sp", version_base=None)
def main(cfg: DictConfig) -> None:
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
        train_dataset, valid_dataset, test_dataset = cifar.get_CIFAR_datasets(
            validation_ratio=cfg["dataset"]["validation_ratio"],
            dataset_name=dataset_name,
        )
    else:
        raise ValueError("Dataset name must be 'mnist', 'cifar10' or 'cifar100'.")

    batch_size = cfg["dataset"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model_name = cfg["model"]["name"]
    if model_name == "toy-cnn":
        model = SPLayerWiseCNN(cfg=cfg)
    elif model_name.startswith("resnet"):
        model = SPLayerWiseResNet(cfg=cfg, model_name=model_name)
    elif model_name.startswith("vgg"):
        model = SPLayerWiseVGG(cfg=cfg, model_name=model_name)
    else:
        raise ValueError("Model {} is not supported.".format(model_name))
    model.to(device)

    # Optimizer
    optimizer = cfg["optimizer"]["name"]
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError("Optimizer {} is not supported.".format(optimizer))

    # Learning rate scheduling
    lr_scheduler_name = cfg["lr_scheduler"]["name"]
    if lr_scheduler_name == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg["lr_scheduler"]["milestones"],
            gamma=cfg["lr_scheduler"]["gamma"],
        )
    else:
        raise ValueError(
            "Learning rate scheduler {} is not supported.".format(lr_scheduler_name)
        )

    # Set the path to save the trained model
    model_save_path = os.path.join(
        os.getcwd(), "save/layer_wise_sp_model/{}/".format(dataset_name)
    )
    model_save_name = "{}_{}_lr_{}_decay_{}_bsz_{}".format(
        cfg["loss_type"],
        cfg["model"]["name"],
        cfg["optimizer"]["learning_rate"],
        cfg["optimizer"]["weight_decay"],
        cfg["dataset"]["batch_size"],
    )
    model_save_folder = os.path.join(
        model_save_path, model_save_name, cfg["id"], "trial_{}".format(cfg["trial"])
    )
    if not os.path.isdir(model_save_folder):
        os.makedirs(model_save_folder)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(
        logging.FileHandler(os.path.join(model_save_folder, "log.txt"), mode="w")
    )
    logger.info("Model information: {}".format(model))
    with open(os.path.join(model_save_folder, "hyperparameter.txt"), mode="w") as f:
        pprint.pprint(cfg.__dict__, f)

    model.train()
    for epoch in range(cfg["num_epochs"]):
        layer_wise_losses = AverageMeter()
        classification_losses = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            forward_result: SPForwardResult = model.forward_with_signal(x, y)
            for loss in forward_result.layer_wise_loss_list:
                loss.backward()
                layer_wise_losses.update(loss.item(), x.size(0))
            forward_result.classification_loss.backward()
            classification_losses.update(
                forward_result.classification_loss.item(), x.size(0)
            )
            optimizer.step()
        lr_scheduler.step()

        logger.info(
            {
                "epoch": epoch,
                "layer-wise loss": [
                    loss.item() for loss in forward_result.layer_wise_loss_list
                ],
                "layer-wise average loss": layer_wise_losses.avg,
                "classification loss": classification_losses.avg,
            }
        )

        if epoch % cfg["save_freq"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_save_folder, "ckpt_epoch_{}.pth".format(epoch)),
            )

    train_acc = 100.0 * calc_accuracy(
        model=model,
        loader=train_loader,
        device=device,
    )
    test_acc = 100.0 * calc_accuracy(
        model=model,
        loader=test_loader,
        device=device,
    )
    logger.info({"train_acc": train_acc, "test_acc": test_acc})

    torch.save(model.state_dict(), os.path.join(model_save_folder, "last.pt"))
    logger.info({"Model saved to": model_save_folder})


if __name__ == "__main__":
    main()
