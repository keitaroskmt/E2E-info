import torch
from torch import Tensor


class NormalizedHSIC:
    def __init__(self):
        self.epsilon = 1e-5

    def calc_loss(self, x: Tensor, z: Tensor) -> Tensor:
        m = x.size(0)
        k_x = self._gaussian_kernel(x)
        k_z = self._gaussian_kernel(z)

        I = torch.eye(m).to(x.device)
        k_xi = torch.inverse(k_x + self.epsilon * m * I)
        k_zi = torch.inverse(k_z + self.epsilon * m * I)
        r_x = torch.matmul(k_x, k_xi)
        r_z = torch.matmul(k_z, k_zi)
        return torch.sum(r_x * r_z.T)

    def _gaussian_kernel(self, x: Tensor, sigma: float = 5.0) -> Tensor:
        x = x.view(x.size(0), -1)
        dist = torch.norm(x[:, None, :] - x[None, :, :], dim=2)
        gram = torch.exp(-dist / (2 * sigma * sigma * x.size(1)))
        centering = (torch.eye(x.size(0)) - torch.ones(x.size(0), x.size(0)) / x.size(0)).to(x.device)
        return torch.matmul(gram, centering)
