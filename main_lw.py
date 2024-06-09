import os
import logging
import pprint

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf

from datasets import mnist, cifar
from util import calc_accuracy, AverageMeter
from models.layer_wise_model import ModelForwardResult, LayerWiseModel
from models.layer_wise_model_spec import LayerWiseResNetSpec, LayerWiseVGGSpec


@hydra.main(config_path="conf", config_name="main_lw", version_base=None)
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
    valid_loader = DataLoader(
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
    if lr_scheduler_name == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=cfg["lr_scheduler"]["milestones"], gamma=cfg["lr_scheduler"]["gamma"]
        )
    else:
        raise ValueError("Learning rate scheduler {} is not supported.".format(lr_scheduler_name))

    # Set the path to save the trained model
    model_save_path = os.path.join(os.getcwd(), "save/layer_wise_model/{}/".format(dataset_name))
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
    best_valid_acc = 0.0
    test_acc = 0.0
    best_epoch = 0
    for epoch in range(cfg["num_epochs"]):
        layer_wise_losses = AverageMeter()
        classification_losses = AverageMeter()

        for x, y in train_loader:
            optimizer.zero_grad()
            if cfg["loss_type"] == "supervised_contrastive":
                x1, x2, y = x[0].to(device), x[1].to(device), y.to(device)
                model_forward_result: ModelForwardResult = model.forward_with_loss_aug(x1, x2, y)
            else:
                x, y = x.to(device), y.to(device)
                model_forward_result: ModelForwardResult = model.forward_with_loss(x, y)
            for loss in model_forward_result.layer_wise_loss_list:
                loss.backward()
                layer_wise_losses.update(loss.item(), y.size(0))
            model_forward_result.classification_loss.backward()
            classification_losses.update(model_forward_result.classification_loss.item(), y.size(0))
            optimizer.step()
        lr_scheduler.step()

        # Validation and Logging
        if cfg["dataset"]["validation_ratio"] > 0.0 and cfg["loss_type"] != "supervised_contrastive":
            valid_acc = 100.0 * calc_accuracy(model=model, loader=valid_loader, device=device)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(model_save_folder, "best.pth"))
                test_acc = 100.0 * calc_accuracy(model=model, loader=test_loader, device=device)
            logger.info(
                {
                    "epoch": epoch,
                    "valid_acc": valid_acc,
                    "layer-wise average loss": layer_wise_losses.avg,
                    "classification loss": classification_losses.avg,
                }
            )
        else:
            logger.info(
                {
                    "epoch": epoch,
                    "layer-wise average loss": layer_wise_losses.avg,
                    "classification loss": classification_losses.avg,
                }
            )
        # Save the trained model
        if epoch % cfg["save_freq"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_save_folder, "ckpt_epoch_{}.pth".format(epoch)),
            )

    if cfg["loss_type"] != "supervised_contrastive":
        train_acc = 100.0 * calc_accuracy(model=model, loader=train_loader, device=device)
        if cfg["dataset"]["validation_ratio"] > 0.0:
            logger.info({"Last train_acc": train_acc, "best epoch": best_epoch, "test_acc": test_acc})
        elif not "contrastive" in cfg["loss_type"]:
            test_acc = 100.0 * calc_accuracy(model=model, loader=test_loader, device=device)
            logger.info({"Last train_acc": train_acc, "Last test_acc": test_acc})

    # Save the last model
    torch.save(model.state_dict(), os.path.join(model_save_folder, "last.pth"))

    pprint.pprint(cfg)
    print("Output saved to {}".format(model_save_folder))


if __name__ == "__main__":
    main()
