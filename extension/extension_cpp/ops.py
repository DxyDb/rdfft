import torch
from torch import Tensor


__all__ = ["myrfft", "myirfft"] 

def myrfft(a: Tensor) -> Tensor:
    return torch.ops.extension_cpp.myrfft1.default(a)

def myirfft(a: Tensor) -> Tensor:
    return torch.ops.extension_cpp.myirfft1.default(a)
