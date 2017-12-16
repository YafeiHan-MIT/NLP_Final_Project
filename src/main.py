#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:13:21 2017

@author: yafei
"""
# import other function modules
import argparse
import torch
import datetime
import cPickle as pickle
import pdb
import sys
import os

from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
os.chdir(dirname(dirname(realpath(__file__)))) #u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

from src.data_util import *
from src.model import get_model
from src.train_util import train_model

#Build a argument parser: argparser
argparser = argparse.ArgumentParser(description="Neural network for QA")
argparser.add_argument("--corpus",
        type = str
    )
argparser.add_argument("--train",
        type = str,
        default = "data/train_random.txt"
    )
argparser.add_argument("--test",
        type = str,
        default = "data/test.txt"
    )
argparser.add_argument("--dev",
        type = str,
        default = "data/dev.txt"
    )
argparser.add_argument("--embeddings",
        type = str,
        default = ""
    )
argparser.add_argument("--hidden_dim", "-d",
        type = int,
        default = 200
    )
argparser.add_argument("--learning",
        type = str,
        default = "adam"
    )
argparser.add_argument("--learning_rate",
        type = float,
        default = 0.001
    )
argparser.add_argument("--l2_reg",
        type = float,
        default = 1e-5
    )
argparser.add_argument("--activation", "-act",
        type = str,
        default = "tanh"
    )
argparser.add_argument("--batch_size",
        type = int,
        default = 40
    )
argparser.add_argument("--depth",
        type = int,
        default = 1
    )
argparser.add_argument("--dropout",
        type = float,
        default = 0.0
    )
argparser.add_argument("--max_epoch",
        type = int,
        default = 20
    )
argparser.add_argument("--cut_off",
        type = int,
        default = 1
    )
argparser.add_argument("--max_seq_len",
        type = int,
        default = 100
    )
argparser.add_argument("--normalize",
        type = int,
        default = 1
    )
argparser.add_argument("--reweight",
        type = int,
        default = 1
    )
argparser.add_argument("--order",
        type = int,
        default = 2
    )
argparser.add_argument("--model_name",
        type = str,
        default = "lstm"
    )
argparser.add_argument("--mode",
        type = int,
        default = 1
    )
argparser.add_argument("--outgate",
        type = int,
        default = 0
    )
argparser.add_argument("--load_pretrain",
        type = str,
        default = ""
    )
argparser.add_argument("--average",
        type = int,
        default = 1
    )
argparser.add_argument("--save_model",
        type = str,
        default = "result_lstm"
    )

argparser.add_argument("--if_save",
        type = int,
        default = 1
    )

argparser.add_argument("--margin",
        type = float,
        default = 0.2
    )

        
argparser.add_argument("--seed",
        type = int,
        default = 7
    )

argparser.add_argument("--cuda",
        type = bool,
        default = False
    )

argparser.add_argument("--pad_left",
        type = bool,
        default = False
    )

argparser.add_argument("--bidirectional",
        type = bool,
        default = True
    )

args = argparser.parse_args()
print args
print ""

#torch.manual_seed(args.seed)

if not os.path.exists(args.save_model):
    os.makedirs(args.save_model)

if __name__ == '__main__':
    print("\nLoad dataset:")
    embedding_path = 'data/vector/vectors_pruned.200.txt.gz'
    embeddings, word_to_indx = getEmbeddingTable(embedding_path)
    print "vocab size:", embeddings.shape[0] - 1
    print "embed dim:", embeddings.shape[1]

    ##read raw corpus data
    corpus_path = 'data/text_tokenized.txt.gz'
    raw_corpus = read_corpus(corpus_path)
    print "corpus size:", len(raw_corpus)  # 167765

    ##convert raw_corpus to ids_corpus
    ids_corpus = map_corpus(raw_corpus, embeddings, word_to_indx, max_len=100)

    ###read annotation data
    train = read_annotations(args.train, num_neg=20)

    dev_path = 'data/dev.txt'
    dev = read_annotations(args.dev, num_neg=20)
    test_path = 'data/test.txt'
    test = read_annotations(args.test, num_neg=20)

    # print "num of training queries:", len(train)
    # print "number of dev queries:", len(dev) ##189
    # print "number of test queries:", len(test) ##186
    
    ##Load model 
    model = get_model(embeddings, args, ids_corpus)
    print(model)

    if args.cuda:
        model = model.cuda()
    # train
    if args.mode == 1: #training 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
        train_model(ids_corpus, model, train, dev, test)


