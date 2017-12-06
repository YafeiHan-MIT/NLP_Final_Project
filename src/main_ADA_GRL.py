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
sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
os.chdir(dirname(dirname(realpath(__file__)))) #u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

import src.init_util 
from src.data_util import *
from src.model_lstm_ADA_GRL import *
from src.train_util_ADA_GRL import *

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
        default = "data/vector/glove_word_vectors.txt.gz"
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
        default = "lstm_ada"
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
        default = "result_lstm_ada"
    )

argparser.add_argument("--if_save",
        type = int,
        default = 0
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

argparser.add_argument("--lambd",
        type = float,
        default = 0.7
    )

argparser.add_argument("--hidden_dim_dc",
        type = float,
        default = 200
    )

argparser.add_argument("--padding_id",
       type = int,
       default = 0
    )

argparser.add_argument("--max_unchanged",
       type = int,
       default = 15
    )

argparser.add_argument("--hidden_layers",
       type = int,
       default = 1
    )

args = argparser.parse_args()
print args
print ""


torch.manual_seed(args.seed)

if __name__ == '__main__':
    print "\nLoading soruce and target corpus.."
    source_corpus_path = 'data/text_tokenized.txt.gz'
    source_corpus = read_corpus(source_corpus_path)
    print '\tsize of source corpus:',  len(source_corpus)
    
    target_corpus_path = 'data/Android/corpus.tsv.gz'
    target_corpus = read_corpus(target_corpus_path)
    print '\tsize of target corpus:',  len(target_corpus)
    
    print '\nLoading embedding lookup table..'
    embeddings, word_to_indx = getEmbeddingTable(args.embeddings)
    print "\tvocab size:", embeddings.shape[0]-1
    print "\tembed dim:", embeddings.shape[1]
    
    print '\nConvert raw corpus to word ids'
    src_corpus_ids = map_corpus(source_corpus, embeddings, word_to_indx, max_len=args.max_seq_len)
    tar_corpus_ids= map_corpus(target_corpus, embeddings, word_to_indx, max_len=args.max_seq_len)
    args.src_corpus_ids = src_corpus_ids
    args.tar_corpus_ids = tar_corpus_ids
    
    print "\nRead annotations from source domain (train)" 
    train = read_annotations(args.train, num_neg=20) 
    print "\tnum of training queries:", len(train)
    
    print "\nRead annotations from target domain (dev/test)" 
    tar_dev = read_annotations_target('data/Android/dev.pos.txt','data/Android/dev.neg.txt') #a tuple of 2: (pos_pairs,neg_pairs)
    tar_test = read_annotations_target('data/Android/test.pos.txt','data/Android/test.neg.txt') #a tuple of 2: (pos_pairs,neg_pairs)
    
    ##Initialize model and optimizers
    model = LSTM_ADA(embeddings, args)                        
    #Initializing optimizer: 
    optimizer_f = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    optimizer_d= torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)

    #optimizer_f = torch.optim.Adam([p for p in model.parameters()][1:5], lr = args.learning_rate, weight_decay=0) 
    #optimizer_d= torch.optim.Adam([p for p in model.parameters()][5:], lr = -args.learning_rate, weight_decay=0) 
    train_model(model, train, tar_dev, tar_test, optimizer_f,optimizer_d)


###Try run one epoch
#result = run_epoch(train, tar_dev, tar_test, model, optimizer_f,optimizer_d, args)

###test evaluate
#auc,auc005= model.evaluate_auc(tar_dev)

###Generate one example batch
#N = len(src_batches)
#train_loss_y = 0.0
#train_cost_y = 0.0  
#train_loss_d = 0.0 
#
#i=0
#src_titles,src_bodies,triples=src_batches[i]
#tar_titles,tar_bodies = tar_batches[i]
#titles = np.hstack([src_titles,tar_titles])
#bodies = np.hstack([src_bodies,tar_bodies])
#num_ques = titles.shape[1]
#batch = (titles,bodies,triples)
#
#
###test forward step by step 
##model = get_model(embeddings, args)
##seq_len_t, num_ques= titles.shape
##seq_len_b, num_ques= bodies.shape
##model.hcn_t = model.init_hidden(num_ques) #a tuple (h_0, c_0) for titles' initial hidden states (1 * num_ques * hidden_dim)
##model.hcn_b = model.init_hidden(num_ques) #(h_0, c_0) for bodies' initial hidden states
##
##titles = Variable(torch.from_numpy(titles).long(), requires_grad=False)
##bodies = Variable(torch.from_numpy(bodies).long(), requires_grad=False)
##
##embeds_titles = model.embedding(titles) #seq_len_title * num_ques * embed_dim
##embeds_bodies = model.embedding(bodies)
##
#### lstm layer: word embedding (200) & h_(t-1) (hidden_dim) => h_t (hidden_dim)
##h_t, model.hcn_t = model.lstm(embeds_titles, model.hcn_t)
##h_b, model.hcn_b = model.lstm(embeds_bodies, model.hcn_b)
##
#### activation function 
##h_t = model.activation(h_t) #seq_len * num_ques * hidden_dim
##h_b = model.activation(h_b) #seq_len * 
##
###if args.normalize:
##h_t = normalize_3d(h_t)
##h_b = normalize_3d(h_b)
##
##if model.args.average: # Average over sequence length, ignoring paddings
##    h_t_final = model.average_without_padding(h_t, titles) #h_t: num_ques * hidden_dim
##    h_b_final = model.average_without_padding(h_b, bodies) #h_b: num_ques * hidden_dim
##h_final = (h_t_final+h_b_final)*0.5 # num_ques * hidden_dim
###h_final = apply_dropout(h_final, dropout) ???
##h_final = normalize_2d(h_final) ##normalize along hid
##
##h=gradient_reverse(h_final,model.lambd) #hidden feature after GRL,  input for domain classification
##h = model.linear1(h)
##h = model.activation(h)
##o = model.linear2(h)
##o = model.log_softmax(o)
#
###forward only in one line 
#model = get_model(embeddings, args)
#h_final,domain_pred=model(batch)  #h_final: feature extractor outcome; pred_d: predicted domain (log_softmax)
#
#print h_final
###Evaluate losses 
#domain_true = torch.zeros(num_ques)
#domain_true[num_ques/2:]=1 
#domain_true = autograd.Variable(domain_true.long())
#
###--------------test backpropagation
#
#h_final_src = h_final[:num_ques/2,:]  #first half is from src domain 
#loss_y = max_margin_loss(args,h_final_src,triples,args.margin)
#cost_y = loss_y + model.get_l2_reg()
#
###backpropgate domain classification loss
#loss_d = F.nll_loss(domain_pred,domain_true)
#
###backprop
#f_ext = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
#d_clf= torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
#
#model.train()
#par0 = deepcopy([p.data for p in model.parameters()])
#print par0[1][0:5,:]
#print par0[6][0:5]
##gr0 = deepcopy([p.grad for p in model.parameters()])
##print gr0
#
#f_ext.zero_grad()
#loss_y.backward(retain_graph=True)  #back propagation, compute gradient 
#f_ext.step() 
#
#par1 = deepcopy([p.data for p in model.parameters()])
#print par1[1][0:5,:] #changed 
#print par1[6][0:5] #not changed 
##gr1 = deepcopy([p.grad for p in model.parameters()])
##print gr1
#
#f_ext.zero_grad()
#d_clf.zero_grad()
#loss_d.backward()  #back propagation, compute gradient 
#f_ext.step() 
#d_clf.step()
#par2 = deepcopy([p.data for p in model.parameters()])
#print par2[1][0:5,:]
#print par2[6][0:5] #not changed 
##gr2 = deepcopy([p.grad for p in model.parameters()])
##print gr2
#
#print model(batch)[0]
#
#train_loss_y += loss_y.data
#train_cost_y += cost_y.data
#train_loss_d += loss_d.data


