# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
def _default_name(obj_class):
    return obj_class.__name__
def register(name=None, registry=None):
    def decorator(fn, registration_name=None):
        module_name = registration_name or _default_name(fn)
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn
    return lambda fn: decorator(fn, name)
from . import Linear

registry = {}
register = partial(register, registry=registry)
class SFU(torch.nn.Module):
    """
    only two input, one input vector and one fusion vector

    Args:
        - input_size:
        - fusions_size:
    Inputs:
        - input: (seq_len, batch, input_size)
        - fusions: (seq_len, batch, fusions_size)
    Outputs:
        - output: (seq_len, batch, input_size)
    """

    def __init__(self, input_size, fusions_size):
        super(SFU, self).__init__()

        self.linear_r = torch.nn.Linear(input_size * 4, input_size)
        self.linear_g = torch.nn.Linear(input_size * 4, input_size)

    def forward(self, input, fusions):
        # print('input:',input.shape)
        # print('fusions:',fusions.shape)
        m = torch.cat((input, fusions, input * fusions, input - fusions), dim=-1)
        r = f.tanh(self.linear_r(m))  # (seq_len, batch, input_size)
        g = f.sigmoid(self.linear_g(m))  # (seq_len, batch, input_size)
        o = g * r + (1 - g) * input

        return o

@register('simple')
class Fusion(nn.Module):
    def __init__(self,  input_size):
        super().__init__()
        self.fusion = Linear(input_size * 2, input_size, activations=True)

    def forward(self, x, align):
        return self.fusion(torch.cat([x, align], dim=-1))


@register('full')
class FullFusion(nn.Module):
    def __init__(self,  input_size):
        super().__init__()
        self.dropout = 0.2
        hidden_size=input_size
        self.sfu=SFU(input_size,hidden_size)
        self.fusion1 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion2 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion3 = Linear(input_size * 2, hidden_size, activations=True)
        self.fusion = Linear(hidden_size * 3, hidden_size, activations=True)
        self.fusion0 = Linear(input_size, hidden_size, activations=True)

    def forward(self, x, align):
        # x1 = self.fusion1(torch.cat([x, align], dim=-1))
        # x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
        # x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
        # x = torch.cat([x1, x2, x3], dim=-1)
        # x = f.dropout(x, self.dropout, self.training)
        y=self.sfu(x,align)
        x = f.dropout(y, self.dropout, self.training)
        return self.fusion0(x)
