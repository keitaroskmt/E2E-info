import torch
from torch import Tensor


class NormalizedHSIC:
    """
    Class for computing the normalized HSIC score between two sets of representations.
    Currently, Gaussian and linear kernels are supported.
    """

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = device
        self.epsilon = 1e-5

    def calc_score(
        self,
        x: Tensor,
        y: Tensor,
        x_kernel: str = "gaussian",
        y_kernel: str = "gaussian",
    ) -> Tensor:
        batch_size = x.size(0)
        k_x = self._kernel_matrix(x, kernel=x_kernel)
        k_y = self._kernel_matrix(y, kernel=y_kernel)

        m_I = torch.eye(batch_size).to(self.device)
        k_x_inv = torch.inverse(k_x + self.epsilon * batch_size * m_I)
        k_y_inv = torch.inverse(k_y + self.epsilon * batch_size * m_I)
        r_x = torch.matmul(k_x, k_x_inv)
        r_y = torch.matmul(k_y, k_y_inv)
        return torch.sum(r_x * r_y.T)

    def _kernel_matrix(self, x: Tensor, kernel: str = "gaussian") -> Tensor:
        if kernel == "gaussian":
            return self._gaussian_kernel(x)
        elif kernel == "linear":
            return self._linear_kernel(x)
        else:
            raise ValueError("Unknown kernel type: {}".format(kernel))

    def _gaussian_kernel(self, x: Tensor, sigma: float = 5.0) -> Tensor:
        x = x.view(x.size(0), -1)
        dist = torch.norm(x[:, None, :] - x[None, :, :], dim=2)
        gram = torch.exp(-(dist**2) / (2 * sigma * sigma * x.size(1)))
        centering = (
            torch.eye(x.size(0)) - torch.ones(x.size(0), x.size(0)) / x.size(0)
        ).to(self.device)
        return torch.matmul(gram, centering)

    def _linear_kernel(self, x: Tensor) -> Tensor:
        return torch.matmul(x, x.T)
