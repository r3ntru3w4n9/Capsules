r"""
Stateless functions are provided here,
as opposed to stateful functions (modules)

:def: `init`

* apply initialization of the modules recursively

:def: `squash`

* as in paper, squash a vector into probability (length in(0, 1)) and state (direction)

:def: `one_hot`

* converts a discrete tensor into one hot encoded vectors

:def: `isnan`

* checking if `nan` appeared in backprop operations

:def: `terminate_on_nan`

* `raise SystemExit` on discovery of `nan`

:def: `to_device`

* move a collection of tensors onto the target device recursively
"""
import torch
from torch import nn
from torch.utils.data import Dataset


def init(module: nn.Module):
    try:
        nn_init.xavier_normal_(
            tensor=module.weight.data, gain=nn_init.calculate_gain("relu")
        )
    except AttributeError:
        pass


def squash(x: torch.Tensor) -> torch.Tensor:
    norm2 = (x ** 2).sum()
    return (x * norm2) / (torch.sqrt(input=norm2) * (1 + norm2))


def one_hot(x: torch.Tensor, categories: int) -> torch.Tensor:
    zeros = x.new_zeros(size=(*x.size(), categories), dtype=torch.float)
    return zeros.scatter_(
        dim=-1,
        index=x.unsqueeze(dim=-1),
        src=x.new_ones(zeros.shape, dtype=torch.float),
    )


def isnan(tensor: torch.Tensor or None):
    if tensor is None:
        return False
    else:
        return torch.isnan(tensor).any()


def terminate_on_nan(module: nn.Module, grad_input: tuple, grad_output: tuple):
    grad_input = (isnan(g) for g in grad_input)
    grad_output = (isnan(g) for g in grad_output)
    if any(grad_input) or any(grad_output):
        print("NaN encountered")
        raise SystemExit


def to_device(iterable, device: str):
    if isinstance(iterable, (list, tuple, Dataset)):
        return tuple(to_device(iterable=i, device=device) for i in iterable)
    elif isinstance(iterable, torch.Tensor):
        return iterable.to(device)
    else:
        return torch.tensor(iterable, device=device)
