"""Main file for sequential layer-wise training.
Each layer is trained on the top of fully trained preceding layers, and the model is trained in a sequential manner.
"""

import os
import logging
import pprint

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

from src.datasets import mnist, cifar
from src.util import calc_accuracy, AverageMeter
from src.models.layer_wise_model import ModelForwardResultSpecifiedLayer, LayerWiseModel
from src.models.layer_wise_model_spec import LayerWiseResNetSpec, LayerWiseVGGSpec


@hydra.main(config_path="conf", config_name="main_lw_seq", version_base=None)
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
        if cfg["loss_type"] == "supervised_contrastive":
            if cfg["model"]["name"].startswith("vgg"):
                train_dataset, valid_dataset, test_dataset = cifar.get_CIFAR_supcon_datasets(
                    validation_ratio=cfg["dataset"]["validation_ratio"],
                    dataset_name=dataset_name,
                    train_input_size=32,
                )
            else:
                train_dataset, valid_dataset, test_dataset = cifar.get_CIFAR_supcon_datasets(
                    validation_ratio=cfg["dataset"]["validation_ratio"],
                    dataset_name=dataset_name,
                )
                cfg["dataset"]["size"] = 28
        else:
            train_dataset, valid_dataset, test_dataset = cifar.get_CIFAR_datasets(
                validation_ratio=cfg["dataset"]["validation_ratio"],
                dataset_name=dataset_name,
            )
    else:
        raise ValueError("Dataset name must be 'mnist', 'cifar10' or 'cifar100'.")

    batch_size = cfg["dataset"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg["num_workers"], pin_memory=True
    )
    # Validation currently is not supported in sequential layer-wise training.
    _ = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )

    # Model
    model_name = cfg["model"]["name"]
    if model_name.startswith("resnet"):
        model_spec = LayerWiseResNetSpec(cfg=cfg)
    elif model_name.startswith("vgg"):
        model_spec = LayerWiseVGGSpec(cfg=cfg)
    else:
        raise ValueError("Model {} is not supported.".format(model_name))
    model = LayerWiseModel(model_spec=model_spec, num_classes=cfg["dataset"]["num_classes"])
    model.to(device)

    # Optimizer
    optimizer_name = cfg["optimizer"]["name"]
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["optimizer"]["learning_rate"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["optimizer"]["learning_rate"], weight_decay=cfg["optimizer"]["weight_decay"]
        )
    else:
        raise ValueError("Optimizer {} is not supported.".format(optimizer_name))

    # Learning rate scheduling
    lr_scheduler_name = cfg["lr_scheduler"]["name"]

    # Set the path to save the trained model
    model_save_path = os.path.join(os.getcwd(), "save/layer_wise_model_sequentially/{}/".format(dataset_name))
    model_save_name = "{}_{}_lr_{}_decay_{}_bsz_{}_head_{}".format(
        cfg["loss_type"],
        cfg["model"]["name"],
        cfg["optimizer"]["learning_rate"],
        cfg["optimizer"]["weight_decay"],
        cfg["dataset"]["batch_size"],
        cfg["head_type"],
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
    torch.autograd.set_detect_anomaly(True)
    test_acc = 0.0

    assert hasattr(model, "num_trainable_layers")
    for layer_index in range(model.num_trainable_layers):
        # Initialize learning rate scheduler every time a new layer is trained
        if lr_scheduler_name == "multisteplr":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=cfg["lr_scheduler"]["milestones"], gamma=cfg["lr_scheduler"]["gamma"]
            )
        else:
            raise ValueError("Learning rate scheduler {} is not supported.".format(lr_scheduler_name))

        losses = AverageMeter()
        for epoch in range(cfg["num_epochs"]):
            for x, y in train_loader:
                optimizer.zero_grad()
                if cfg["loss_type"] == "supervised_contrastive":
                    x1, x2, y = x[0].to(device), x[1].to(device), y.to(device)
                    model_forward_result: ModelForwardResultSpecifiedLayer = model.forward_with_loss_aug_sequentially(x1, x2, y, layer_index)
                else:
                    x, y = x.to(device), y.to(device)
                    model_forward_result: ModelForwardResultSpecifiedLayer = model.forward_with_loss_sequentially(x, y, layer_index)
                loss = model_forward_result.loss
                loss.backward()
                losses.update(loss.item(), y.size(0))
                optimizer.step()
            lr_scheduler.step()

            logger.info(
                {
                    "epoch": epoch,
                    "layer_index": layer_index,
                    "loss": losses.avg,
                }
            )
            # Save the trained model
            if epoch % cfg["save_freq"] == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_save_folder, "ckpt_layer_index_{}_epoch_{}.pth".format(layer_index, epoch)),
                )

    if cfg["loss_type"] != "supervised_contrastive":
        train_acc = 100.0 * calc_accuracy(model=model, loader=train_loader, device=device)
        if not "contrastive" in cfg["loss_type"]:
            test_acc = 100.0 * calc_accuracy(model=model, loader=test_loader, device=device)
            logger.info({"Last train_acc": train_acc, "Last test_acc": test_acc})

    # Save the last model
    torch.save(model.state_dict(), os.path.join(model_save_folder, "last.pth"))

    pprint.pprint(cfg)
    print("Output saved to {}".format(model_save_folder))


if __name__ == "__main__":
    main()
