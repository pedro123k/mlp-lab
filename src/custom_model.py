import torch
import torch.nn as nn
from typing import Optional, List

class CustomModel(nn.Module):

    _transforms_funcs = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'identity': nn.Identity
    }

    def __init__(self, input_size: int, layers: List[int], act_fn: Optional[str] = 'relu', act_out: Optional[str] = 'identity') -> None:
        super(CustomModel, self).__init__()
        self._layers = nn.ModuleList()

        for i in range(len(layers)):
            if i == 0:
                self._layers.append(nn.Linear(input_size, layers[i]))
                self._layers.append(CustomModel._transforms_funcs[act_fn]())
                self._layers.append(nn.Dropout(0.2))
            elif i < len(layers) - 1:
                self._layers.append(nn.Linear(layers[i-1], layers[i]))
                self._layers.append(CustomModel._transforms_funcs[act_fn]())
                self._layers.append(nn.Dropout(0.2))
            else:
                self._layers.append(nn.Linear(layers[i-1], layers[i]))
                self._layers.append(CustomModel._transforms_funcs[act_out]())

    def forward(self, x):
        a = x
        for layer in self._layers:
            a = layer(a)
        return a