import torch
from torch import Tensor
from torch.nn import functional as F

from hsic.hsic import NormalizedHSIC


class LayerWiseLossConfig:
    """
    Configuration for loss functions used in each layer.
    Args:
        cfg: Hydra configuration dictionary.
    """

    def __init__(self, cfg: dict):
        self.loss_type: str = cfg["loss_type"]
        self.num_classes: int = cfg["num_classes"]
        self.nhsic_reg: str = cfg["nhsic_reg"]
        # Settings for supervised contrastive loss.
        self.temperature: float = cfg["temperature"]
        self.base_temperature: float = cfg["base_temperature"]
        # Settings for nHSIC regularization.
        self.nhsic: NormalizedHSIC = NormalizedHSIC()
        self.lambda_nhsic: float = cfg["lambda_nhsic"]

        if not self.loss_type in ["cross_entropy", "similarity", "supervised_contrastive"]:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if not self.nhsic_reg in ["none", "global", "local"]:
            raise ValueError(f"Invalid nHSIC regularization type: {self.nhsic_reg}")


class LayerWiseLoss:
    """
    Loss functions used in the layer-wise training.
    Args:
        loss_cfg: Configuration for loss functions.
    """

    def __init__(self, loss_cfg: LayerWiseLossConfig):
        self.loss_cfg: LayerWiseLossConfig = loss_cfg

    def criterion(self, fx: Tensor, y: Tensor, x: Tensor, model_input: Tensor) -> Tensor:
        """
        Compute loss function.
        Args:
            fx: Output of the block.
            y: Target labels.
            x: Input of the block.
            model_input: Input of the model.
        """

        # Calculate loss.
        if self.loss_cfg.loss_type == "cross_entropy":
            loss = F.cross_entropy(fx, y)
        elif self.loss_cfg.loss_type == "similarity":
            loss = self._similarity_loss(fx, y)
        elif self.loss_cfg.loss_type == "supervised_contrastive":
            loss = self._supervised_contrastive_loss(fx, y)
        else:
            assert False, "Unreachable"

        # Add nHSIC regularization.
        if self.loss_cfg.nhsic_reg == "global":
            loss -= self._calc_nhsic_global(fx, model_input)
        elif self.loss_cfg.nhsic_reg == "local":
            loss -= self._calc_nhsic_local(fx, x)

        return loss

    def _similarity_loss(self, fx: Tensor, y: Tensor) -> Tensor:
        """
        Compute loss function based on "local-loss" paper (https://arxiv.org/abs/1901.06656).
        Args:
            fx: Output of the block.
            y: Target labels.
        """
        y_onehot = F.one_hot(y, num_classes=self.loss_cfg.num_classes).float()
        target_similarity = y_onehot @ y_onehot.T

        # Calculate similarity between samples following https://github.com/anokland/local-loss/blob/master/train.py#L1409
        if fx.dim() == 4:
            if fx.size(1) > 3 and fx.size(2) > 1:
                z = fx.view(fx.size(0), fx.size(1), -1)
                fx = z.std(dim=2)
            else:
                fx = fx.view(fx.size(0), -1)
        fx = fx - fx.mean(dim=1).unsqueeze(1)
        fx = fx / (torch.sqrt(torch.sum(fx**2, dim=1) + 1e-12)).unsqueeze(1)
        output_similarity = (fx @ fx.T).clamp(-1, 1)
        return F.mse_loss(output_similarity, target_similarity)

    def _supervised_contrastive_loss(self, fx: Tensor, y: Tensor) -> Tensor:
        """
        Compute loss function based on "Supervised Contrastive Learning" paper (https://arxiv.org/abs/2004.11362).
        Args:
            fx: Output of the block.
            y: Target labels.
        """
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py
        if fx.dim() > 2:
            fx = fx.view(fx.size(0), -1)
        # Insert normalization
        fx = F.normalize(fx, dim=1)

        batch_size = y.shape[0]
        assert fx.size(0) == 2 * batch_size
        y = y.contiguous().view(-1, 1)
        mask = torch.eq(y, y.T).float().to(y.device)

        # `contrast_feature`: [2 * batch_size, feature_dim]
        contrast_count = 2
        contrast_feature = fx
        anchor_count = contrast_count
        anchor_feature = contrast_feature

        # compute logits
        anchor_dot_product = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_product, dim=1, keepdim=True)
        logits = anchor_dot_product - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases (set the diagonal elements of mask to zero)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(y.device), 0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # Inside log can be zero when the elements of `logits` are very small and the sum of a certain row of `exp_logits` is very small.
        # assert torch.all(exp_logits.sum(1, keepdim=True) > 0.0).item()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def _calc_nhsic_global(self, fx: Tensor, model_input: Tensor) -> Tensor:
        """
        Calculate nHSIC(fx, model_input).
        It is used to prevent the information loss in the block.
        """
        return self.loss_cfg.lambda_nhsic * self.loss_cfg.nhsic.calc_loss(fx, model_input)

    def _calc_nhsic_local(self, fx: Tensor, x: Tensor) -> Tensor:
        """
        Calculate nHSIC(fx, x).
        It is used to prevent the information loss in the block.
        """
        return self.loss_cfg.lambda_nhsic * self.loss_cfg.nhsic.calc_loss(fx, x)
