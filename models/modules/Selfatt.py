import torch
import  math
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import TimeDistributed
from torch.nn import Parameter
from  allennlp.modules.elmo import Elmo
import  numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import os

import torch.nn as nn
import torch.nn.functional as F
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
class TriLinearAttention(nn.Module):
    """
    This function is taken from Allen NLP group, refer to github:
    https://github.com/chrisc36/allennlp/blob/346e294a5bab1ec0d8f2af962cfe44abc450c369/allennlp/modules/tri_linear_attention.py

    TriLinear attention as used by BiDaF, this is less flexible more memory efficient then
    the `linear` implementation since we do not create a massive
    (batch, context_len, question_len, dim) matrix
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self._x_weights = Parameter(torch.Tensor(input_dim, 1))
        self._y_weights = Parameter(torch.Tensor(input_dim, 1))
        self._dot_weights = Parameter(torch.Tensor(1, 1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self.input_dim * 3 + 1))
        self._y_weights.data.uniform_(-std, std)
        self._x_weights.data.uniform_(-std, std)
        self._dot_weights.data.uniform_(-std, std)

    def forward(self, matrix_1, matrix_2):
        # pylint: disable=arguments-differ

        # Each matrix is (batch_size, time_i, input_dim)
        batch_dim = matrix_1.shape[0]
        time_1 = matrix_1.shape[1]
        time_2 = matrix_2.shape[1]

        # (batch * time1, dim) * (dim, 1) -> (batch * time1, 1)
        x_factors = torch.matmul(matrix_1.resize(batch_dim * time_1, self.input_dim), self._x_weights)
        x_factors = x_factors.contiguous().view(batch_dim, time_1, 1)  # ->  (batch, time1, 1)

        # (batch * time2, dim) * (dim, 1) -> (batch * tim2, 1)
        y_factors = torch.matmul(matrix_2.resize(batch_dim * time_2, self.input_dim), self._y_weights)
        y_factors = y_factors.contiguous().view(batch_dim, 1, time_2)  # ->  (batch, 1, time2)

        weighted_x = matrix_1 * self._dot_weights  # still (batch, time1, dim)

        matrix_2_t = torch.transpose(matrix_2, 1, 2)  # -> (batch, dim, time2)

        # Batch multiplication,
        # (batch, time1, dim), (batch, dim, time2) -> (batch, time1, time2)
        dot_factors = torch.matmul(weighted_x, matrix_2_t)

        # Broadcasting will correctly repeat the x/y factors as needed,
        # result is (batch, time1, time2)
        return dot_factors + x_factors + y_factors
class SelfAtt(nn.Module):
    """
        The self attention layer implemented by ourselves, with the function TimeDistributed provided by Allen NLP.
        The self attention get the attention score of cotext and context.
        Refer to : https://allenai.github.io/allennlp-docs/api/allennlp.modules.time_distributed.html?highlight=time%20distributed#module-allennlp.modules.time_distributed
        and to : Attention is all you need pytroch implementation :https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
        Args:
            hidden_size (int): Size of hidden activations.
            drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, hidden_size, drop_prob,i):
        super(SelfAtt, self).__init__()

        self.drop_prob = drop_prob
        if i==1:
            self.att_wrapper = TimeDistributed(nn.Linear(hidden_size * 4, hidden_size))
            self.trilinear = TriLinearAttention(hidden_size)
            self.self_att_upsampler = TimeDistributed(nn.Linear(hidden_size * 3, hidden_size * 4))
            self.enc = nn.GRU(hidden_size, hidden_size // 2, 1,
                              batch_first=True,
                              bidirectional=True)
            self.hidden_size = hidden_size
        else:
            self.att_wrapper = TimeDistributed(nn.Linear(1324 , hidden_size))
            self.trilinear = TriLinearAttention(hidden_size)
            self.self_att_upsampler = TimeDistributed(nn.Linear(hidden_size * 3, 1324 ))
            self.enc = nn.GRU(hidden_size, hidden_size//2 , 1,
                              batch_first=True,
                              bidirectional=True)
            self.hidden_size = hidden_size

    def forward(self, att, c_mask):
        # (batch_size, c_len, 1600)
        att_copy = att.clone()  # To save the original data of attention from pervious layer.
        # (batch_size * c_len, 1600)
        att_wrapped = self.att_wrapper(
            att)  # unroll the second dimention with the first dimension, and roll it back, change of dimension.
        # non-linearity activation function
        att = F.relu(att_wrapped)  # (batch_size * c_len, 1600)
        #         print("att", att.shape)
        c_mask = c_mask.unsqueeze(dim=2).float()  # (batch_size, c_len, 1)

        drop_att = F.dropout(att, self.drop_prob, self.training)  # (batch_size * c_len, hidden_size)
        #         c_mask = c_mask.permute(1, 0, 2)
        #         print(drop_att.shape, c_mask.shape)

        encoder, _ = self.enc(drop_att)
        #         encoder = self.get_similarity_matrix(drop_att, c_mask)
        #         print("encoder", encoder.shape)
        #         encoder = encoder.unsqueeze(dim=3)
      #  print(encoder.shape)
        self_att = self.trilinear(encoder, encoder)  # get the self attention (batch_size, c_len, c_len)

        # to match the shape of the attention matrix
        mask = (c_mask.view(c_mask.shape[0], c_mask.shape[1], 1) * c_mask.view(c_mask.shape[0], 1, c_mask.shape[1]))
        identity = torch.eye(c_mask.shape[1], c_mask.shape[1]).cuda().view(1, c_mask.shape[1], c_mask.shape[1])
        mask = mask * (1 - identity)

        # get the self attention vector features
        self_att_softmax = masked_softmax(self_att, mask, log_softmax=False)
        self_att_vector = torch.matmul(self_att_softmax, encoder)

        # concatenate to make the shape (batch, c_len, 1200)
        conc = torch.cat((self_att_vector, encoder, encoder * self_att_vector), dim=-1)
        #         print("conc", conc.shape)

        # To match with the input attention, we have to upsample the hidden_size from 1200 to 1600.
        upsampler = self.self_att_upsampler(conc)
        out = F.relu(upsampler)

        # (batch_size, c_len, 1600)
        att_copy += out

        att = F.dropout(att_copy, self.drop_prob, self.training)
        return att
class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        key=query
        value=query
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x.cuda()