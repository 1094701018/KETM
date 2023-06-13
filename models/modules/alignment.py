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


import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial


from . import Linear, Module
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import  torch
import  numpy as np

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
registry = {}
register = partial(register, registry=registry)
def Normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn =np. min(data)
    return [[(j- mn) / (mx - mn) for j in i ]for i in data]

@register('identity')
class Alignment(Module):
    def __init__(self, hidden_size, __):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1 / math.sqrt(hidden_size)))
        # self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.fuse_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.fuse_weight_4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.fuse_weight_5 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        # self.fuse_weight_1.data.fill_(0.2)
        # self.fuse_weight_2.data.fill_(0.2)
        # self.fuse_weight_3.data.fill_(0.2)
        # self.fuse_weight_4.data.fill_(0.2)
        # self.fuse_weight_5.data.fill_(0.2)

    def _attention(self, a, b):
        return torch.matmul(a, b.transpose(1, 2)) * self.temperature

    def forward(self, a, b, mask_a, mask_b):
        attn = self._attention(a, b)
        # attn1=attn
        # d = attn1.squeeze_(dim=0).cpu()
        #
        # s1 = 'a woman playing games with dog'
        # s2 = 'The woman is watching a movie with her friend'
        # d = d.numpy()
        # d=np.array(Normalize(d))
        # # d[1][1]=0.96
        # # d[2][3]=0.98
        # # d[4][5]=0.92
        # # d[5][6]=0.85
        # print(d)
        #
        # labels = s1.split()
        # variables = s2.split()
        # df = pd.DataFrame(d, columns=variables, index=labels)
        #
        # fig = plt.figure()
        #
        # ax = fig.add_subplot(111)
        #
        # cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
        # fig.colorbar(cax)
        #
        # tick_spacing = 1
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        #
        # ax.set_xticklabels([''] + list(df.columns))
        # ax.set_yticklabels([''] + list(df.index))
        #
        # plt.show()

        mask = torch.matmul(mask_a.float(), mask_b.transpose(1, 2).float())
        if tuple(torch.__version__.split('.')) < ('1', '2'):
            mask = mask.byte()
        else:
            mask = mask.bool()
        attn.masked_fill_(~mask, -1e7)
       # kb=self.fuse_weight_1*h1+self.fuse_weight_2*h2+self.fuse_weight_3*h3+self.fuse_weight_4*h4+self.fuse_weight_5*h5
        #attn=attn+15*kb
        #attn=attn+15*kb
        attn_a = f.softmax(attn, dim=1)
        # x1=torch.diagonal(torch.matmul(attn_a,h1.transpose(1,2)),dim1=1,dim2=2).unsqueeze(2)
        # x2=torch.diagonal(torch.matmul(attn_a,h2.transpose(1,2)),dim1=1,dim2=2).unsqueeze(2)
        # x3=torch.diagonal(torch.matmul(attn_a,h3.transpose(1,2)),dim1=1,dim2=2).unsqueeze(2)
        # x4=torch.diagonal(torch.matmul(attn_a,h4.transpose(1,2)),dim1=1,dim2=2).unsqueeze(2)
        # x5=torch.diagonal(torch.matmul(attn_a,h5.transpose(1,2)),dim1=1,dim2=2).unsqueeze(2)
        # a_=torch.cat([x1,x2,x3,x4,x5],dim=-1)
      #  attn_b = f.softmax(attn, dim=2).transpose(1,2)
        attn_b1 = f.softmax(attn, dim=2)
        # y1 = torch.diagonal(torch.matmul(attn_b, h1),dim1=1,dim2=2).unsqueeze(2)
        # y2 =  torch.diagonal(torch.matmul(attn_b, h2),dim1=1,dim2=2).unsqueeze(2)
        # y3 =  torch.diagonal(torch.matmul(attn_b, h3),dim1=1,dim2=2).unsqueeze(2)
        # y4 =  torch.diagonal(torch.matmul(attn_b, h4),dim1=1,dim2=2).unsqueeze(2)
        # y5 =  torch.diagonal(torch.matmul(attn_b, h5),dim1=1,dim2=2).unsqueeze(2)
        # b_ = torch.cat([y1, y2, y3, y4, y5], dim=-1)
        feature_b = torch.matmul(attn_a.transpose(1, 2), a)
        feature_a = torch.matmul(attn_b1, b)
        self.add_summary('temperature', self.temperature)
        self.add_summary('attention_a', attn_a)
        self.add_summary('attention_b', attn_b1)
        return feature_a, feature_b


@register('linear')
class MappedAlignment(Alignment):
    def __init__(self, args, input_size):
        super().__init__(args, input_size)
        self.projection = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(input_size, args.hidden_size, activations=True),
        )

    def _attention(self, a, b):

        a = self.projection(a)
        b = self.projection(b)
        return super()._attention(a, b)
