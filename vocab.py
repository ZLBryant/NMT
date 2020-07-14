#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    vocab.py --train-data=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-data=<file>         File of training source sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

import argparse
from utils import pad_sents, read_corpus, read_en_cn_corpus
import torch
from collections import Counter
from itertools import chain
import json
from docopt import docopt

class VocabEntry(object):
    def __init__(self, word2id=None):
        if word2id != None:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0
            self.word2id['<s>'] = 1
            self.word2id['</s>'] = 2
            self.word2id['<unk>'] = 3
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def id2word(self, word_idx):
        return self.id2word[word_idx]

    def add(self, word):
        if word in self:
            return self[word]
        else:
            word_idx = len(self)
            self.word2id[word] = word_idx
            self.id2word[word_idx] = word
            return word_idx

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in sent] for sent in sents]
        return [self[w] for w in sents]

    def indices2words(self, word_idxs, remove_pad=True):
        if remove_pad:
            sent = []
            for idx in word_idxs:
                if idx == self.word2id["</s>"]:
                    break
                sent.append(self.id2word[idx])
            return sent
        return [self.id2word[idx] for idx in word_idxs]

    def to_input_tensor(self, sents):
        idxs = self.words2indices(sents)
        idxs = pad_sents(idxs, self['<pad>'])
        idxs = torch.tensor(idxs, dtype=torch.long)
        return idxs

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

class Vocab(object):
    def __init__(self, src_vocab, trg_vocab):
        self.src = src_vocab
        self.tgt = trg_vocab

    @staticmethod
    def build(src_sents, trg_sents, vocab_size, freq_threshold=2):
        src_vocab = VocabEntry().from_corpus(src_sents, vocab_size, freq_threshold)
        trg_vocab = VocabEntry().from_corpus(trg_sents, vocab_size, freq_threshold)
        return Vocab(src_vocab, trg_vocab)

    def save(self, file_path):
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id), open(file_path, 'w'), indent=2)

    def load(file_path):
        entry = json.load(open(file_path, 'r'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']
        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

def args_init():
    parser = argparse.ArgumentParser(description="Neural MT")
    parser.add_argument("--en_cn", action="store_true")
    parser.add_argument("--train_data", type=str, default="dataset/train.txt", help="train file")
    parser.add_argument("--en_es", action="store_true")
    parser.add_argument("--train_src_data", type=str, default="en_es_data/train.es", help="train src file")
    parser.add_argument("--train_tgt_data", type=str, default="en_es_data/train.en", help="train tgt file")
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--word_freq_threshold", type=int, default=2)
    parser.add_argument("--source_language", type=str, default='en')
    parser.add_argument("--vocab_file_path", type=str, default='vocab.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_init()
    if args.en_cn:
        src_sents, tgt_sents = read_en_cn_corpus(args.train_data, source=args.source_language)
    elif args.en_es:
        src_sents = read_corpus(args.train_src_data, False)
        tgt_sents = read_corpus(args.train_tgt_data, True)
    else:
        print("invalid input!")
        exit(0)
    vocab = Vocab.build(src_sents, tgt_sents, int(args.vocab_size), int(args.word_freq_threshold))
    vocab.save(args.vocab_file_path)