# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS ,SEP= '[PAD]', '[CLS]','[SEP]'  # padding符号, bert中综合信息符号


def build_dataset(config):

    def load_dataset(path, pad_size):
        contents = []
        count=0

        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.rstrip()
                count=count+1
                if not lin:
                    continue

                content1,content2, label = lin.split('\t')
                token = config.tokenizer(content1,content2)
               # token2 = config.tokenizer.tokenize(content2)
                #token = [CLS] + token1+[SEP]+token2
               # seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids= token_ids[:pad_size]
                        seq_len = pad_size
                contents.append((token_ids, int(label), seq_len,mask))
        return contents
    #def load_dataset(path, pad_size):
    #     contents = []
    #
    #
    #     with open(path, 'r', encoding='UTF-8') as f:
    #         for line in tqdm(f):
    #             lin = line.rstrip()
    #
    #             if not lin:
    #                 continue
    #
    #             content1,content2, label = lin.split('\t')
    #             token1 = config.tokenizer.tokenize(content1)
    #             token2 = config.tokenizer.tokenize(content2)
    #             token = [CLS] + token1+[SEP]+token2+[SEP]
    #             seq_len = len(token)
    #             mask = []
    #             token_ids = config.tokenizer.convert_tokens_to_ids(token1)
    #             token2 = config.tokenizer.tokenize(content2)
    #             token2 = [CLS] + token2
    #             seq_len2 = len(token2)
    #             mask2 = []
    #             token_ids2 = config.tokenizer.convert_tokens_to_ids(token2)
    #
    #             if pad_size:
    #                 if len(token1) < pad_size:
    #                     mask1 = [1] * len(token_ids1) + [0] * (pad_size - len(token1))
    #                     token_ids1 += ([0] * (pad_size - len(token1)))
    #                 else:
    #                     mask1 = [1] * pad_size
    #                     token_ids1 = token_ids1[:pad_size]
    #                     seq_len1 = pad_size
    #             if pad_size:
    #                 if len(token2) < pad_size:
    #                     mask2 = [1] * len(token_ids2) + [0] * (pad_size - len(token2))
    #                     token_ids2 += ([0] * (pad_size - len(token2)))
    #                 else:
    #                     mask2 = [1] * pad_size
    #                     token_ids2 = token_ids2[:pad_size]
    #                     seq_len2 = pad_size
    #             contents.append((token_ids1,token_ids2, int(label), seq_len1,seq_len2,mask1, mask2))
    #     return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
     #   x2 = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
       # seq_len2 = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
      #  mask2 = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

import pandas as pd

def mapToNumber(gold_label):
    if gold_label == 'entailment':
        return 0
    if gold_label == 'neutral':
        return 1
    if gold_label == 'contradiction':
        return 2
    return -1

def openData(path):
    data = []
    i = 0

    with open(path,encoding='utf-8') as f:

        for line in f:
            text1, text2, ka1,kb1,label = line.rstrip().split('\t')
           # text1, text2,label = line.rstrip().split('\t')
            if ka1=='':
                ka1=text1
            if kb1=='':
                kb1=text2
            kb=ka1+kb1


            data.append({
                'premise':text1,
                'hypothesis':text2,
                'kb':kb,

                'label':int(label),
            })

    df = pd.DataFrame(data)
    return df

def removeMinVal(df):
    # print(df['label'].unique())
    # Remove data with label -1 (undefined)
    new_df = df[df.label != -1]
    new_df.reset_index(drop=True, inplace=True)
    # print(new_df['label'].unique())
    return new_df


