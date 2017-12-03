#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:05:10 2017

@author: yafei
"""

import torch
from torch import autograd
import numpy as np
from prettytable import PrettyTable
import sys
import time

#from os.path import dirname, realpath
#sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
#os.chdir(dirname(dirname(realpath(__file__)))) #u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

from src.data_util import *
from src.model_lstm import *
from src.init_util import *

def train_model(model, train, dev, test, ids_corpus):
    '''
    Input: 
        train_data: 
        dev_data: 
        model: a sublass of torch.nn.Module
    '''
#    if args.cuda:
#        model = model.cuda()
    args=model.args
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
    dev_eva = []
    test_eva = []
    result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                                    ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
    unchanged = 0
    best_dev_MRR = -1
    start_time = 0
    max_epoch = args.max_epoch
    for epoch in xrange(max_epoch):
        unchanged += 1
        if unchanged > 15: break
        start_time = time.time()
        train_loss, train_cost, dev_eva, test_eva = run_epoch(ids_corpus,train,dev,test, model, optimizer, args)        
        dev_MRR = dev_eva[1]
        if  dev_MRR > best_dev_MRR:
            best_dev_MRR = dev_eva[1]
            unchanged = 0
            result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in list(dev_eva) + list(test_eva)])
            if args.if_save:
                torch.save(model, os.path.join(args.save_model,"model_ep"+str(epoch)+".pkl.gz"))
            say("\r\n\n")

            type((time.time()-start_time)/60.0)
            say(( "Epoch {}\tcost={:.3f}\tloss={:.3f}\tdev_MRR={:.2f}\tTime:{:.3f}m\n").format(
                    epoch,
                    train_cost,
                    train_loss,
                    dev_MRR,
                    (time.time()-start_time)/60.0
            ))
            say("\tp_norm: {}\n".format(
                    model.get_pnorm_stat()
                ))
            say("\n")
            say("{}".format(result_table))
            say("\n")
    torch.save(model, os.path.join(args.save_model,"model_ep"+str(epoch)+"_best.pkl.gz"))


def run_epoch(ids_corpus, train, dev, test, model, optimizer, args):
    '''
    Run one epoch (one pass of data)
    Return average training loss, training cost, dev and test evaluations after this epoch. 
    '''
    #set model to training mode 
    model.train() #set model to train mode 
    train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id=0, perm=None, pad_left=args.pad_left)
    #N=3 ##for testing 
    N = len(train_batches)
    train_loss = 0.0
    train_cost = 0.0     
    for i in xrange(N):
        batch=train_batches[i]
        #print "batch title..", batch[0][0]
        h_final=model(batch,True)
        loss = max_margin_loss(args,h_final,batch,args.margin)
        cost = loss + model.get_l2_reg()
        
        ##check params updates  
        #params1 = [p.data for p in model.parameters()][1]
        #print "params1:", params1
        
        ##backprop
        optimizer.zero_grad()
        loss.backward()  #back propagation, compute gradient 
        optimizer.step() #update parameters 
        
        train_loss += loss.data
        train_cost += cost.data
        if i%10 == 0:
            say("\r{}/{}".format(i,N))
            print "  loss, cost:", (train_loss/(i+1))[0], (train_cost/(i+1))[0]
        dev_eva = None
        test_eva = None
        if i == N-1: #last batch 
            if dev is not None:
                #MAP_dev, MRR_dev, P1_dev, P5_dev = model.evaluate(dev)
                dev_eva = model.evaluate(dev)
            if test is not None:
                #MAP_test, MRR_test, P1_test, P5_test = model.evaluate(test)   
                test_eva = model.evaluate(test)
    return (train_loss/(i+1))[0], (train_cost/(i+1))[0], dev_eva, test_eva

        
#embedding_path='data/vector/vectors_pruned.200.txt.gz'
#embeddings, word_to_indx = getEmbeddingTable(embedding_path)
#print "vocab size:", embeddings.shape[0]-1
#print "embed dim:", embeddings.shape[1]
#
###read raw corpus data 
#corpus_path='data/text_tokenized.txt.gz'
#raw_corpus=read_corpus(corpus_path)
#print "corpus size:", len(raw_corpus) #167765
#
###convert raw_corpus to ids_corpus
#ids_corpus = map_corpus(raw_corpus, embeddings, word_to_indx,max_len=100)
#
####read annotation data 
#anoPath = 'data/train_random.txt'
#train = read_annotations(anoPath, num_neg=20)
##nt "num of training queries:", len(train)
#
##Create dev batches 
#dev_path = 'data/dev.txt'
#dev = read_annotations(dev_path, num_neg=20)
##dev_batches = create_eval_batches(ids_corpus, dev, padding_id=0, pad_left=False)
##print "number of dev queries:", len(dev) ##189
#
###Create test batches 
#test_path = 'data/test.txt'
#test = read_annotations(test_path, num_neg=20)
##nt "number of test queries:", len(test) ##186
#
###run 1 epoch
##model = get_model(embeddings, ids_corpus, args)
##optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
##train_loss, train_cost, dev_eva, test_eva = run_epoch(ids_corpus, train_batches, dev, test, model, optimizer, args)
#
###Run multi epochs
#model = get_model(embeddings, ids_corpus, args)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)
#train_model(model, train, dev, test, ids_corpus)

