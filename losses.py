r"""
Loss functions used for `CapsNet`

:class: `MarginLoss`

* implements the function
$$Loss = T_c max(0, m^+-|v_c|)^2 + \lambda (1- T_c) max(0, |v_c|-m^-)^2$$

:class: `ReconLoss`

* uses MSE loss (the reason for the existence of the class is for
 conveniently swapping to BCE loss,
 which turned out not to be effective for learning capsule networks)
"""
import torch
from torch.nn import functional as F


class MarginLoss(object):
    def __init__(self, T_c: float, lmbda: float, boundary: tuple, categories: int):
        self._T_c = T_c
        self._lmbda = lmbda
        (self._top, self._bottom) = boundary
        self._categories = categories

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        v_c = torch.sqrt_((input ** 2).sum(dim=1))

        one_hot = self.one_hot(target, self._T_c, categories=self._categories)

        return (
            (
                (one_hot * F.relu(self._top - v_c) ** 2)
                + ((1 - one_hot) * self._lmbda * F.relu(v_c - self._bottom) ** 2)
            )
            .sum(dim=-1)
            .mean()
        )

    @staticmethod
    def one_hot(x: torch.Tensor, T_c: float, categories: int) -> torch.Tensor:
        zeros = x.new_full(
            size=(*x.size(), categories), fill_value=1 - T_c, dtype=torch.float
        )
        return zeros.scatter_(
            dim=-1,
            index=x.unsqueeze(dim=-1),
            src=x.new_full(size=zeros.shape, fill_value=T_c, dtype=torch.float),
        )


class ReconLoss(object):
    def __call__(self, input, target):
        return F.mse_loss(input=input, target=target.view(*input.size()))
