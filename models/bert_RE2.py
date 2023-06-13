# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction
from .modules.BiDAFAttention import BiDAFAttention
from .modules.Selfatt import SelfAtt
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/scitail/train.txt'                                # 训练集
        self.dev_path = dataset + '/scitail/dev.txt'                                    # 验证集
        self.test_path = dataset + '/scitail/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/scitail/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 6000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                             # epoch数
        self.batch_size = 32                                        # mini-batch大
        self.pad_size = 96                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 256
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(config, config.hidden_size if i == 0 else 100),
            'alignment': alignment["linear"](
                config, config.hidden_size + 200 if i == 0 else 100),
            'fusion': fusion["full"](
                config, config.hidden_size + 200 if i == 0 else 100),
        }) for i in range(1)])

        self.att = BiDAFAttention(hidden_size=200, drop_prob=0.2)
        self.self_att = SelfAtt(hidden_size=200,
                            drop_prob=0.2, i=1)
        self.pooling = Pooling()
        self.prediction = prediction['full'](config)
    def forward(self, x):
        context1 = x[0]  # 输入的句子
        context2 = y[0]  # 输入的句子
        mask1 = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        mask2 = y[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        a, text_cls1 = self.bert(context1, attention_mask=mask1, output_all_encoded_layers=False)
        b, text_cls2 = self.bert(context2, attention_mask=mask2, output_all_encoded_layers=False)
        mask1=mask1.unsqueeze(2)>0
        mask2=mask2.unsqueeze(2)>0
        for i, block in enumerate(self.blocks):
            if i > 0:
                break;
            #
            #     a = self.connection(a, res_a, i)
            #     b = self.connection(b, res_b, i)
            #     res_a, res_b = a, b
            # print('a',a.shape)
            # print('mask_a',mask_a.shape)
            a_enc = block['encoder'](a, mask1)
            b_enc = block['encoder'](b, mask2)
            # print('ae:',a_enc.shape)
            # print('be:',b_enc.shape)
            # self_a_enc = self.self_att1(a_enc, mask_a)
            # self_b_enc=self.self_att1(b_enc,mask_b)
            # a=self_a_enc+res_a
            # b=self_b_enc+res_b
            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)

            align_a, align_b = block['alignment'](a, b, mask1, mask2)
            # print('ka',ka.shape)
            # print(ka)
            # print('kb',kb.shape)
            # print(kb)
            # print('a:',a.shape)
            # print('ala:',align_a.shape)
            # # print('b:',b.shape)
            # print('blb:',align_b.shape)
            # print('ka',ka.shape)
            # print('kb',kb.shape)
            # align_a=torch.cat([align_a,ka],dim=-1)
            # align_b=torch.cat([align_b,kb],dim=-1)
            # a=torch.cat([a,ka],dim=-1)
            # b=torch.cat([b,kb],dim=-1)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)
            # a=self.linear(a)
            # b=self.linear(b)
            # print('a:',a.shape)
            # print('b:',b.shape)
        # a=torch.cat([a,ka],dim=-1)
        # b=torch.cat([b,kb],dim=-1)

        att = self.att(a, b, mask1, mask2)  # p a
        # print("att:",att.shape)
        att1 = self.self_att(att, mask1)
        # print('att1:',att1.shape)
        # print(mask_a.shape)
        # print(mask_a)
        att3 = self.pooling(att1, mask1)

        # #  mod, _ = self.mod(att1)
        #   mod1=mod.transpose(0,1)
        #   att3 = torch.cat((mod1[0], mod1[-1]), dim=-1)
        # mod2 = self.pooling(mod, mask_a)
        y = self.prediction(att3)
        return y