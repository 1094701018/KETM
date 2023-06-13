import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from transformers import BertTokenizer,BertModel
#from pytorch_pretrained import BertModel


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

import random

import jsonlines
import  os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
from utils import openData, removeMinVal
from transformers import ElectraModel, ElectraTokenizer
import logging
from typing import Collection
import math
from models.modules import Module,ModuleList,ModuleDict
from models.modules.encoder import Encoder
from models.modules.fusion import registry as fusion
from models.modules.alignment import registry as alignment
from models.modules.connection import registry as connection
from models.modules.pooling import Pooling
from models.modules.prediction import registry as prediction
from models.modules.fusion import Linear
class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 2
def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs
# class SFU(torch.nn.Module):
#     """
#     only two input, one input vector and one fusion vector
#
#     Args:
#         - input_size:
#         - fusions_size:
#     Inputs:
#         - input: (seq_len, batch, input_size)
#         - fusions: (seq_len, batch, fusions_size)
#     Outputs:
#         - output: (seq_len, batch, input_size)
#     """
#
#     def __init__(self, input_size, fusions_size):
#         super(SFU, self).__init__()
#
#         self.linear_r = torch.nn.Linear(input_size * 4, input_size)
#         self.linear_g = torch.nn.Linear(input_size * 4, input_size)
#
#     def forward(self, input, fusions):
#         # print('input:',input.shape)
#         # print('fusions:',fusions.shape)
#         m = torch.cat((input, fusions, input * fusions, input - fusions), dim=-1)
#         r = F.tanh(self.linear_r(m))  # (seq_len, batch, input_size)
#         g = F.sigmoid(self.linear_g(m))  # (seq_len, batch, input_size)
#         o = g * r + (1 - g) * input
#
#         return o
class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.
    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).
    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
#         batch_size, c_len, _ = c.size()
#         q_len = q.size(1)
#         s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
#         c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
#         q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
#         s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
#         s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

#         # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
#         a = torch.bmm(s1, q)
#         # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
#         b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

#         x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)

       # s=s+15*kb.transpose(1,2)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c* a,c*b], dim=2)  # (bs, c_len, 4 * hid_size)


        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).
        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.
        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        # print(c.shape)
        # print(self.c_weight.shape)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s
# class FullFusion(nn.Module):
#     def __init__(self, input_size):
#         super().__init__()
#         self.dropout = 0.2
#         self.sfu=SFU(input_size,input_size)
#         # self.fusion1 = Linear(input_size * 2, args.hidden_size, activations=True)
#         # self.fusion2 = Linear(input_size * 2, args.hidden_size, activations=True)
#         # self.fusion3 = Linear(input_size * 2, args.hidden_size, activations=True)
#         # self.fusion = Linear(args.hidden_size * 3, args.hidden_size, activations=True)
#         #self.fusion0 = Linear(input_size, input_size, activations=True)
#
#     def forward(self, x, align):
#         # x1 = self.fusion1(torch.cat([x, align], dim=-1))
#         # x2 = self.fusion2(torch.cat([x, x - align], dim=-1))
#         # x3 = self.fusion3(torch.cat([x, x * align], dim=-1))
#         # x = torch.cat([x1, x2, x3], dim=-1)
#         # x = f.dropout(x, self.dropout, self.training)
#         y=self.sfu(x,align)
#         x = F.dropout(y, self.dropout, self.training)
#         return x
# class Pooling(nn.Module):
#     def forward(self, x, mask):
#         mask = mask.unsqueeze(2).bool()
#         return x.masked_fill_(~mask.bool(), -float('inf')).max(dim=1)[0]
# class AugmentedResidual(nn.Module):
#     def __init__(self, _):
#         super().__init__()
#
#     def forward(self, x, res, i):
#         if i == 1:
#             return torch.cat([x, res], dim=-1)  # res is embedding
#         hidden_size = x.size(-1)
#         x = (res[:, :, :hidden_size] + x) * math.sqrt(0.5)
#         return torch.cat([x, res[:, :, hidden_size:]], dim=-1)  # latter half of res is embedding
# class ModuleList(nn.ModuleList):
#     def get_summary(self, base_name=''):
#         summary = {}
#         if base_name:
#             base_name += '/'
#         for i, module in enumerate(self):
#             if hasattr(module, 'get_summary'):
#                 name = base_name + str(i)
#                 summary.update(module.get_summary(name))
#         return summary
#
#
# class ModuleDict(nn.ModuleDict):
#     def get_summary(self, base_name=''):
#         summary = {}
#         if base_name:
#             base_name += '/'
#         for key, module in self.items():
#             if hasattr(module, 'get_summary'):
#                 name = base_name + key
#                 summary.update(module.get_summary(name))
#         return summary
class SFU1(torch.nn.Module):
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
        super(SFU1, self).__init__()

        self.linear_r = torch.nn.Linear(input_size * 4, input_size)
        self.linear_g = torch.nn.Linear(input_size * 4, input_size)

    def forward(self, input, fusions):
        # print('input:',input.shape)
        # print('fusions:',fusions.shape)
        m = torch.cat((input, fusions, input * fusions, input - fusions), dim=-1)
        r = F.tanh(self.linear_r(m))  # (seq_len, batch, input_size)
        g = F.sigmoid(self.linear_g(m))  # (seq_len, batch, input_size)
        o = g * r + (1 - g) * input

        return o
# class MultiheadAttention(nn.Module):
#     # n_heads：多头注意力的数量
#     # hid_dim：每个词输出的向量维度
#     def __init__(self, hid_dim, n_heads, dropout):
#         super(MultiheadAttention, self).__init__()
#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#
#         # 强制 hid_dim 必须整除 h
#         assert hid_dim % n_heads == 0
#         # 定义 W_q 矩阵
#         self.w_q = nn.Linear(hid_dim, hid_dim)
#         # 定义 W_k 矩阵
#         self.w_k = nn.Linear(hid_dim, hid_dim)
#         # 定义 W_v 矩阵
#         self.w_v = nn.Linear(hid_dim, hid_dim)
#         self.fc = nn.Linear(hid_dim, hid_dim)
#         self.do = nn.Dropout(dropout)
#         # 缩放
#         self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
#
#     def forward(self, query, mask=None):
#         # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
#         # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
#         # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
#         bsz = query.shape[0]
#         key=query
#         value=query
#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)
#         # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
#         # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
#         # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
#         # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
#         # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
#         # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
#         # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
#         Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#         K = K.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#         V = V.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#
#         # 第 1 步：Q 乘以 K的转置，除以scale
#         # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
#         # attention：[64,6,12,10]
#         Q=Q.to(device)
#         K=K.to(device)
#         attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
#
#         # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
#         if mask is not None:
#             attention = attention.masked_fill(mask == 0, -1e10)
#
#         # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
#         # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
#         # attention: [64,6,12,10]
#         attention = self.do(torch.softmax(attention, dim=-1)).to(device)
#         V=V.to(device)
#
#         # 第三步，attention结果与V相乘，得到多头注意力的结果
#         # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
#         # x: [64,6,12,50]
#         x = torch.matmul(attention, V)
#
#         # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
#         # x: [64,6,12,50] 转置-> [64,12,6,50]
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # 这里的矩阵转换就是：把多组注意力的结果拼接起来
#         # 最终结果就是 [64,12,300]
#         # x: [64,12,6,50] -> [64,12,300]
#         x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
#         x = self.fc(x)
#         return x
class Model(nn.Module):

    def __init__(self, bert_path):
        super(Model, self).__init__()
        self.bert1 =BertModel.from_pretrained(bert_path)
        self.bert2 =BertModel.from_pretrained(bert_path)
        # self.bert3 =BertModel.from_pretrained(bert_path)
        # self.bert4 =BertModel.from_pretrained(bert_path)

        #self.bert1 = BertModel.from_pretrained(bert_path)
        self.dim_mlp = 768
        self.hidden=200
        input_emb_size=self.dim_mlp
        for param in self.bert1.parameters():
            param.requires_grad = True
        for param in self.bert2.parameters():
            param.requires_grad = True
        # for param in self.bert3.parameters():
        #     param.requires_grad = True
        # for param in self.bert4.parameters():
        #     param.requires_grad = True
        self.att=BiDAFAttention(self.hidden)
       # self.multi_att = MultiheadAttention(self.hidden, 8, dropout=0.2)
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(self.dim_mlp if i == 0 else self.dim_mlp + self.hidden,self.hidden),
            'alignment': alignment['identity'](
                self.hidden, self.dim_mlp + self.hidden  if i == 0 else input_emb_size + self.hidden * 2),
            'fusion': fusion['full'](
                self.hidden  if i == 0 else input_emb_size + self.hidden * 2),
        }) for i in range(1)])
        self.fc=nn.Linear(4 * self.hidden, self.hidden)
        self.connection = connection['aug']()
        # self.pool = nn.Sequential(nn.Linear(4 * self.dim_mlp, self.dim_mlp),
        #                             self.bert.pooler)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])

        self.dropout = nn.Dropout(0.2)
        self.pool=Pooling()
        self.fusion = SFU1(self.hidden * 4, self.hidden)
        # self.fusion = FullFusion(768)
        # self.dense = nn.Sequential(
        #     nn.Dropout(0.2),
        #     Linear(768 , 768, activations=True),
        #     nn.Dropout(0.2),
        #
        # )
        self.fc_cnn = Linear(self.hidden*4, 3)
        self.fc_cnn1=Linear(768,3)
        self.fc_cnn2=Linear(2*768,3)



    def forward(self,context,mask0, context1,mask1,context2,mask2,kb,kbmask):
        # context = x['input_ids']  # 输入的句子
        # mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
       # kb_hid,cls=self.bert(ids,attention_mask=masks,output_all_encoded_layers=False)
       #hid, pool = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out3 = self.bert1(context, attention_mask=mask0)
        out1= self.bert3(context1, attention_mask=mask1)
        out2 = self.bert4(context2, attention_mask=mask2)
        out4 = self.bert2(kb, attention_mask=kbmask)
        a=out1[0]
        b = out2[0]
        pool=out3[0]
        pool1=out4[0]
        cls=pool[:,0,:]
        kbcls=pool1[:,0,:]
        res_a, res_b = a, b
        mask_a=mask1.unsqueeze(2).bool()
        mask_b=mask2.unsqueeze(2).bool()

        for i,block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)

                res_a, res_b = a, b
            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)


            # a = self.multi_att(a_enc)
            # b = self.multi_att(b_enc)
            a = a_enc
            b = b_enc

            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)


        out=self.att(a,b,mask_a,mask_b)
        out1=self.pool(out,mask_a)

        out0=self.fc_cnn(out1)
        out2=self.fc_cnn1(cls)

        out3=out+out2
        out=torch.cat([cls,kbcls],dim=-1)
        out3_=self.fc_cnn2(out)
        out4=self.fc_cnn1(cls)
        out5=out4+out3_



       # print(hid[:,0,:].shape)

       # pool=out[1]
        # print(out)
        # print(out[0].shape)
        # print(out[1].shape)
        # print(hid)
        # print(hid[0][0])
        # print(pool[0].shape)
       # print(xxx)


     #   hid1, pool1 = self.bert1(context, attention_mask=mask, output_all_encoded_layers=False)
        # a=self.att(hid,kb_hid,mask,masks)
        # b=self.pool(a)
      #  out=self.fusion(pool,cls)
      #  out=torch.cat([pool,out,cls],dim=-1)
       # out=self.dense()
      #  out = self.fc_cnn(hid[:,0,:])
        # encoded_premises = hid
        # encoded_hypotheses =kb_hid
        # attended_premises, attended_hypotheses = self.attention(encoded_premises, mask, encoded_hypotheses,
        #                                                         masks)
        # enhanced_premises = torch.cat([encoded_premises,
        #                                attended_premises,
        #                                encoded_premises - attended_premises,
        #                                encoded_premises * attended_premises], dim=-1)
        # enhanced_hypotheses = torch.cat([encoded_hypotheses, attended_hypotheses,
        #                                  encoded_hypotheses - attended_hypotheses,
        #                                  encoded_hypotheses * attended_hypotheses], dim=-1)
        # projected_premises = self.projection(enhanced_premises)
        # projected_hypotheses = self.projection(enhanced_hypotheses)
        # pair_embeds = torch.cat([projected_premises, projected_hypotheses, projected_premises - projected_hypotheses,
        #                          projected_premises * projected_hypotheses], dim=-1)
        #
        # pair_output = self.pool(pair_embeds)
        # pair_output=torch.cat([cls,pair_output],dim=-1)
       # out=self.out(pair_output)

        return out3+out5
epochs = 60
MAX_LENGTH =128
batch_size =24*3
SEED_VAL = 10
LEARNING_RATE = 2e-5
EPS = 1e-8

# BERT_MODEL = 'microsoft/deberta-base'
# BERT_MODEL = 'roberta-large'
# BERT_MODEL = 'distilbert-base-uncased'
BERT_MODEL = '/home/iip/Jiangkexin/BERT_test/bert_pretrain1'
# BERT_MODEL = 'albert-base-v2'

LOG_PATH = 'log/training-' + 'scibert_base+kb' + 'mednli' + str(epochs) + '-' + str(LEARNING_RATE) + '.log'
logging.basicConfig(filename=LOG_PATH, level=logging.INFO)

label_dict = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2
}

logging.info(' ######### Training Started! #########')


def initial_log():
    logging.info('\n ** Training Configuration **')
    logging.info(f'Model            : {BERT_MODEL}')
    logging.info(f'Epoch            : {epochs}')
    logging.info(f'Batch Size       : {batch_size}')
    logging.info(f'Learning Rate    : {LEARNING_RATE}')
    logging.info(f'EPS              : {EPS}')
    logging.info(f'Seed Value       : {SEED_VAL}\n')


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def evaluate(model, device, dataloader_val, PARALLEL_GPU=False):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)
        labels=batch[8]
        # inputs = {'input_ids': batch[0],
        #           'attention_mask': batch[1],
        #           'labels': batch[2],
        #           }

        with torch.no_grad():
            outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7])

        loss = F.cross_entropy(outputs, labels.long())
        logits = outputs

        if PARALLEL_GPU:
            # Assumed that batch sizes are equally divided across the GPUs.
            loss = loss.mean()

        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    # Hardcoded label
    target_names = ['entailment', 'neutral','conration']
   # target_names = ['entailment', 'neutral']

    cr = classification_report(labels_flat, preds_flat, target_names=target_names, digits=4)
    logging.info(f'\n *** CLASSIFICATION REPORT ***\n\n{cr}\n')

    total_pred = 0
    total_true = 0
    total_acc = 0



    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        logging.info(f'Class: {label_dict_inverse[label]}')
        logging.info(
            f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)} = {len(y_preds[y_preds == label]) / len(y_true)}\n')
        total_pred += len(y_preds[y_preds == label])
        total_true += len(y_true)
        total_acc += (total_pred / total_true)


    logging.info(f'Accuracy : {total_pred}/{total_true} = {(total_pred / total_true)}')
    return total_pred / total_true
    # logging.info(f'Accuracy (equal weight each class): {total_acc/len(label_dict)} \n')


def main():
    a = 0
    b=0
    initial_log()
    print(torch.cuda.is_available())
    # Check GPU Availability
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using GPU!')
        gpu_num = torch.cuda.device_count()
        print(f'There are {gpu_num} GPU available!')
        for i in range(0, gpu_num):
            gpu_name = torch.cuda.get_device_name(i)
            print(f' -- GPU {i + 1}: {gpu_name}')
    else:
        device = torch.device('cpu')
        print('Using CPU :(')

    print('\n')

    # Import Dataset
    print('#### Importing Dataset ####')
    df_train = openData(r'/home/iip/Jiangkexin/BERT_test/data/mednli/train.txt')
    df_test = openData(r'/home/iip/Jiangkexin/BERT_test/data/mednli/test.txt')
    df_val = openData(r'/home/iip/Jiangkexin/BERT_test/data/mednli/dev.txt')
    # df_train=df_train[:10]
    # df_test=df_test[:10]
    # df_val=df_val[:10]
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    df_train = removeMinVal(df_train)
    # train=[]
    # test=[]
    # for i in range(len(df_train)):
    #     a=df_train.premise[i]+'[SEP]'+df_train.hypothesis[i]
    #     train.append(a)
    # for i in range(len(df_test)):
    #     b=df_test.premise[i]+'[SEP]'+df_test.hypothesis[i]
    #     test.append(b)
    df_val = removeMinVal(df_val)
    df_test = removeMinVal(df_test)

    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)

    print('#### Download Tokenizer & Tokenizing ####')

    tokenizer =BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    print('Encoding training data')
    encode_ptrain = tokenizer(df_train.premise.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)
    encode_htrain = tokenizer(df_train.hypothesis.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
    encode_train = tokenizer(df_train.premise.tolist(),df_train.hypothesis.tolist(),
                              return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
    # encode_train1 = tokenizer(train,
    #                          return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)

    encode_test = tokenizer(df_test.premise.tolist(), df_test.hypothesis.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
    encode_val = tokenizer(df_val.premise.tolist(), df_val.hypothesis.tolist(),
                            return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)

    encode_kbtrain = tokenizer(df_train.kb.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)
    encode_kbval = tokenizer(df_val.kb.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)
    encode_kbtest = tokenizer(df_test.kb.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)

    labels_train = torch.tensor(df_train.label.values)
    print('Encoding validation data')
    # encode_val = tokenizer(df_val.premise.tolist(), df_val.hypothesis.tolist(),
    #                        return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)
    encode_pval = tokenizer(df_val.premise.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)
    encode_hval = tokenizer(df_val.hypothesis.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)

    labels_val = torch.tensor(df_val.label.values)

    print('Encoding test data')
    encode_ptest = tokenizer(df_test.premise.tolist(),
                            return_tensors='pt', padding='max_length', max_length=MAX_LENGTH,truncation=True)
    encode_htest = tokenizer(df_test.hypothesis.tolist(),
                             return_tensors='pt', padding='max_length', max_length=MAX_LENGTH, truncation=True)

    labels_test = torch.tensor(df_test.label.values)

    dataset_train = TensorDataset(encode_train['input_ids'],encode_train['attention_mask'],encode_ptrain['input_ids'], encode_ptrain['attention_mask'], encode_htrain['input_ids'], encode_htrain['attention_mask'],encode_kbtrain['input_ids'],encode_kbtrain['attention_mask'],labels_train)
    dataset_val = TensorDataset(encode_val['input_ids'], encode_val['attention_mask'], encode_pval['input_ids'], encode_pval['attention_mask'],encode_hval['input_ids'], encode_hval['attention_mask'],encode_kbval['input_ids'],encode_kbval['attention_mask'],labels_val)
    dataset_test = TensorDataset(encode_test['input_ids'],encode_test['attention_mask'],encode_ptest['input_ids'], encode_ptest['attention_mask'],encode_htest['input_ids'], encode_htest['attention_mask'],encode_kbtest['input_ids'],encode_kbtest['attention_mask'],labels_test)
    # dataset_kbtrain = TensorDataset(encode_kbtrain['input_ids'],encode_kbtrain['attention_mask'],labels_train)
    # dataset_kbtest = TensorDataset(encode_kbtest['input_ids'],encode_kbtest['attention_mask'],labels_test)
    # dataset_kbval = TensorDataset(encode_kbval['input_ids'],encode_kbval['attention_mask'],labels_val)
    # dataset_train = TensorDataset(encode_train['input_ids'], encode_train['attention_mask'],encode_train['token_type_ids'],
    #                                labels_train)
  #  dataset_val = TensorDataset(encode_val['input_ids'], encode_val['attention_mask'], encode_val['token_type_ids'], labels_val)
    # dataset_test = TensorDataset(encode_test['input_ids'], encode_test['attention_mask'], encode_test['token_type_ids'],labels_test)
    print('#### Downloading Pretrained Model ####')
    # model1 = BertForSequenceClassification.from_pretrained(BERT_MODEL,
    #                                                       num_labels=len(label_dict),
    #                                                       output_attentions=False,
    #                                                       output_hidden_states=False)
    model=Model(BERT_MODEL)

    dataloader_train = DataLoader(dataset_train,
                                  sampler=RandomSampler(dataset_train),
                                  batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                       sampler=SequentialSampler(dataset_val),
                                       batch_size=batch_size)

    dataloader_test = DataLoader(dataset_test,
                                 sampler=SequentialSampler(dataset_test),
                                 batch_size=batch_size)

    print('#### Setting Up Optimizer ####')
    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE,
                      eps=EPS)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    # TRAINING

    PARALLEL_GPU = False
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        PARALLEL_GPU = True
    model = model.to(device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total:{}  Trainable:{}' .format(total_num,trainable_num))


    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('#### Training Started! ####')
    for epoch in tqdm(range(1, epochs + 1)):

        model.train()

        loss_train_total = 0

        # dataloader_train
        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        i = 0
        for batch in progress_bar:
            i += 1
            model.zero_grad()

            batch = tuple(b.to(device) for b in batch)
            labels=batch[8]
            # print(batch[0])
            # print(batch[1])
            # print(batch[2])
            # inputs = {'input_ids': batch[0],
            #           'attention_mask': batch[1],
            #           'labels':batch[2],
            #
            #           }

            outputs = model(batch[0],batch[1],batch[2],batch[3],batch[4],batch[5],batch[6],batch[7])
          #  outputs=model(**inputs)

            loss = F.cross_entropy(outputs, labels.long())

            if PARALLEL_GPU:
                # Assumed that batch sizes are equally divided across the GPUs.
                # print('Loss are averaged! Assume that batch sizes are equally divided across the GPUs')
                loss = loss.mean()

            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

        # Save model every epoch
        # torch.save(model.state_dict(), f'./models/deberta2/finetuned_model_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')
        logging.info(f'\n -------- Epoch {epoch} ---------- \n')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        logging.info(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate(model, device, dataloader_validation, PARALLEL_GPU)
        val_f1 = f1_score_func(predictions, true_vals)
        acc1 = accuracy_per_class(predictions, true_vals)
        if acc1>b:
            b=acc1
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        logging.info(f'Validation loss: {val_loss}')
        logging.info(f'F1 Score (Weighted): {val_f1} \n \n')

        # Evaluate per epoch
        # Evaluate validation data
        # logging.info(f' -- Validation Data -- \n')
        # _, predictions, true_vals = evaluate(model, device, dataloader_validation, PARALLEL_GPU)
        # accuracy_per_class(predictions, true_vals)

        logging.info(f' -- Test Data -- \n')
        # Evaluate test data
        _, predictions, true_vals = evaluate(model, device, dataloader_test, PARALLEL_GPU)
        acc=accuracy_per_class(predictions, true_vals)
        if acc>a:
            a=acc
        logging.info(' --MAx TEST ACC -- {}'.format(a))
        logging.info(' --MAx VAL ACC -- {}'.format(b))

    # Save final epoch model
    torch.save(model.state_dict(), f'./models/scibert/finetuned_model_epoch_{epochs}.model')


if __name__ == '__main__':
    print(torch.cuda.is_available())

    main()




