#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:59:07 2017

@author: yafei
"""
import argparse
import torch
import datetime
import cPickle as pickle
import pdb
import sys
import os

from os.path import dirname, realpath

sys.path.append(dirname(dirname(realpath(__file__))))  ##add project path to system path list
os.chdir(
    dirname(dirname(realpath(__file__))))  # u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

from src.train_util_ADA_GRL import *

# Build a argument parser: argparser
argparser = argparse.ArgumentParser(description="Neural network for QA")
argparser.add_argument("--corpus",
                       type=str
                       )
argparser.add_argument("--train",
                       type=str,
                       default="data/train_random.txt"
                       )
argparser.add_argument("--test",
                       type=str,
                       default="data/test.txt"
                       )
argparser.add_argument("--dev",
                       type=str,
                       default="data/dev.txt"
                       )
argparser.add_argument("--hidden_dim", "-d",
                       type=int,
                       default=200
                       )
argparser.add_argument("--learning",
                       type=str,
                       default="adam"
                       )
argparser.add_argument("--learning_rate",
                       type=float,
                       default=0.001
                       )
argparser.add_argument("--l2_reg",
                       type=float,
                       default=1e-5
                       )
argparser.add_argument("--activation", "-act",
                       type=str,
                       default="tanh"
                       )
argparser.add_argument("--batch_size",
                       type=int,
                       default=40
                       )
argparser.add_argument("--depth",
                       type=int,
                       default=1
                       )
argparser.add_argument("--dropout",
                       type=float,
                       default=0.0
                       )
argparser.add_argument("--max_epoch",
                       type=int,
                       default=20
                       )
argparser.add_argument("--cut_off",
                       type=int,
                       default=1
                       )
argparser.add_argument("--max_seq_len",
                       type=int,
                       default=100
                       )
argparser.add_argument("--normalize",
                       type=int,
                       default=1
                       )
argparser.add_argument("--reweight",
                       type=int,
                       default=1
                       )
argparser.add_argument("--order",
                       type=int,
                       default=2
                       )
argparser.add_argument("--model_name",
                       type=str,
                       default="lstm"
                       )
argparser.add_argument("--mode",
                       type=int,
                       default=1
                       )
argparser.add_argument("--outgate",
                       type=int,
                       default=0
                       )
argparser.add_argument("--load_pretrain",
                       type=str,
                       default=""
                       )
argparser.add_argument("--average",
                       type=int,
                       default=1
                       )
argparser.add_argument("--save_model",
                       type=str,
                       default="result_lstm_ada"
                       )

argparser.add_argument("--if_save",
                       type=int,
                       default=0
                       )

argparser.add_argument("--margin",
                       type=float,
                       default=0.2
                       )

argparser.add_argument("--seed",
                       type=int,
                       default=7
                       )

argparser.add_argument("--cuda",
                       type=bool,
                       default=False
                       )

argparser.add_argument("--pad_left",
                       type=bool,
                       default=False
                       )

argparser.add_argument("--lambd",
                       type=float,
                       default=0.7
                       )

argparser.add_argument("--hidden_dim_dc",
                       type=float,
                       default=200
                       )

argparser.add_argument("--padding_id",
                       type=int,
                       default=0
                       )

argparser.add_argument("--max_unchanged",
                       type=int,
                       default=15
                       )

argparser.add_argument("--hidden_layers",
                       type=int,
                       default=1
                       )

args = argparser.parse_args()
print args
print ""

torch.manual_seed(args.seed)

if __name__ == '__main__':
    print "\nLoading source and target corpus.."
    source_corpus_path = 'data/text_tokenized.txt.gz'
    source_corpus, source_vocab = read_corpus_get_unique(source_corpus_path)
    print '\tsize of source corpus:', len(source_corpus)

    print 'Unique words in source: ', len(source_vocab)

    target_corpus_path = 'data/Android/corpus.tsv.gz'
    target_corpus, target_vocab = read_corpus_get_unique(target_corpus_path)
    print '\tsize of target corpus:', len(target_corpus)

    print 'Unique words in target: ', len(target_vocab)
    vocab = source_vocab.union(target_vocab)

    print 'Total number of unique words: ', len(vocab)

    print '\nLoading embedding lookup table..'
    embeddings_path = 'data/vector/glove_full.txt.gz'
    embeddings, word_to_indx = getEmbeddingTable(embeddings_path, vocab)
    print "\tvocab size:", embeddings.shape[0] - 1
    print "\tembed dim:", embeddings.shape[1]

    print '\nConvert raw corpus to word ids'
    src_corpus_ids = map_corpus(source_corpus, embeddings, word_to_indx, max_len=args.max_seq_len)
    tar_corpus_ids = map_corpus(target_corpus, embeddings, word_to_indx, max_len=args.max_seq_len)
    args.src_corpus_ids = src_corpus_ids
    args.tar_corpus_ids = tar_corpus_ids

    print "\nRead annotations from source domain (train)"
    train = read_annotations(args.train, num_neg=20)
    print "\tnum of training queries:", len(train)

    print "\nRead annotations from target domain (dev/test)"
    tar_dev = read_annotations_target('data/Android/dev.pos.txt',
                                      'data/Android/dev.neg.txt')  # a tuple of 2: (pos_pairs,neg_pairs)
    tar_test = read_annotations_target('data/Android/test.pos.txt',
                                       'data/Android/test.neg.txt')  # a tuple of 2: (pos_pairs,neg_pairs)

    # Initialize model and optimizers
    if args.model_name == "lstm":
        model = LSTM_ADA(args, embeddings)
    elif args.model_name == "cnn":
        model = CNN_ADA(args, embeddings)
    else:
        raise Exception("Model name " + args.model_name + " not supported")

    # Initializing optimizer:
    optimizer_f = torch.optim.Adam([p for p in model.parameters()][1:5], lr=args.learning_rate, weight_decay=0)
    optimizer_d = torch.optim.Adam([p for p in model.parameters()][5:], lr=args.learning_rate, weight_decay=0)

    train_model(model, train, tar_dev, tar_test, optimizer_f, optimizer_d)
