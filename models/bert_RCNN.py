# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained import BertModel, BertTokenizer
from models.modules.fusion import FullFusion
from models.modules.BiDAFAttention import BiDAFAttention
import math
class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
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


class Model(nn.Module):

    def __init__(self, bert_path):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.dim_mlp = 768
        for param in self.bert.parameters():
            param.requires_grad = True
        self.att=BiDAFAttention(self.dim_mlp)
        self.fc=nn.Linear(4 * self.dim_mlp, self.dim_mlp)
        self.pool = nn.Sequential(nn.Linear(4 * self.dim_mlp, self.dim_mlp),
                                    self.bert.pooler)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, 256, (k, 768)) for k in (2, 3, 4)])
        self.dropout = nn.Dropout(0.2)
        # self.lstm = nn.GRU(768, 128, 1,
        #                    bidirectional=True, batch_first=True, dropout=0.2)
        # self.lstm1 = nn.GRU(256, 128, 1,
        #                     bidirectional=True, batch_first=True, dropout=0.2)
        # self.lstm2 = nn.GRU(512, 128, 1,
        #                     bidirectional=True, batch_first=True, dropout=0.2)
        self.fusion = FullFusion(768)
        self.dense = nn.Sequential(
            nn.Dropout(0.2),
            Linear(768*3 , 768, activations=True),
            nn.Dropout(0.2),

        )
        self.fc_cnn = Linear(768, 3)
      #  self.attention = SoftmaxAttention()


    def forward(self, context,mask,ids,masks):
        # context = x['input_ids']  # 输入的句子
        # mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        kb_hid,cls=self.bert(ids,attention_mask=masks,output_all_encoded_layers=False)
        # hid, pool = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        # a=self.att(hid,kb_hid,mask,masks)
        # b=self.fc(a)
        # # hid1, _ = self.lstm(hid)  # 256
        # # hid2, _ = self.lstm1(hid1)  # 256
        # # # # # print(hid1.shape)
        # # # # # print(hid2.shape)
        # # hid3, _ = self.lstm2(torch.cat([hid1, hid2], dim=-1))  # 256
        # # # #  print(hid3.shape)
        # # out = torch.cat([hid1, hid2, hid3], dim=-1)
        # # kb_hid = kb_hid.unsqueeze(1)
        # # hid=hid.unsqueeze(1)
        # # # #   print(out.shape)
        # # kb_hid = torch.cat([self.conv_and_pool(kb_hid, conv) for conv in self.convs], 1)
        # # hid = torch.cat([self.conv_and_pool(hid, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        # out1 = self.fusion(cls,pool)
        # out2 = self.fusion(hid,kb_hid)
        # out=self.fusion(pool,cls)
        # out=torch.cat([pool,out,cls],dim=-1)
        # out=self.dense(out)
        out = self.fc_cnn(cls)
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

        return out
