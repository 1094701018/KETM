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
from functools import partial

def register(name=None, registry=None):
    def decorator(fn, registration_name=None):
        module_name = registration_name or _default_name(fn)
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn
    return lambda fn: decorator(fn, name)


def _default_name(obj_class):
    return obj_class.__name__
from . import Linear

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Prediction(nn.Module):
    def __init__(self, args, inp_features=1):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(200 * inp_features, 200, activations=True),
            nn.Dropout(args.dropout),
            Linear(200, args.num_classes),
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


@register('full')
class AdvancedPrediction(Prediction):
    def __init__(self, args):
        super().__init__(args, inp_features=4)

    def forward(self, a):
        return self.dense(a)


@register('symmetric')
class SymmetricPrediction(AdvancedPrediction):
    def forward(self, a, b):
        return self.dense(torch.cat([a, b, (a - b).abs(), a * b], dim=-1))
