import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
import nltk


def pad_sents(sents, pad_token):
    sents_padded = []
    max_len = max([len(sent) for sent in sents])
    for sent in sents:
        sents_padded.append(sent + [pad_token] * max(0, max_len - len(sent)))

    return sents_padded

def read_en_cn_corpus(file_path, source):
    en = []
    cn = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if source == 'cn':
                en.append(['<s>'] + nltk.word_tokenize(line[0]) + ['</s>'])
                cn.append(jieba.lcut(line[1]))
            else:
                en.append(nltk.word_tokenize(line[0]))
                cn.append(['<s>'] + jieba.lcut(line[1]) + ['</s>'])
        if source == 'cn':
            return cn, en
        return en, cn

def read_corpus(file_path, is_target):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if is_target:
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
    return data


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        if i == batch_num - 1:
            a = 1
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
