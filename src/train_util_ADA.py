#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:05:10 2017
@author: yafei
"""

import numpy as np
import sys
import time

import torch
from torch import autograd
import torch.nn.functional as F
from prettytable import PrettyTable

from src.init_util import *
from src.data_util import *
from src.model_lstm_ADA import *
from src.meter import *

def train_model(model, train, tar_dev, tar_test,optimizer_f,optimizer_d):
    '''
    Input: 
        train: training data from source domain  
        tar_dev: dev data from target domain. a tuple of length 2: (pos_pairs,neg_pairs) 
        tar_test: test data from target domain. a tuple of length 2: (pos_pairs,neg_pairs)
    '''

    if model.args.cuda:
        model = model.cuda()

    result_table = PrettyTable(["Epoch", "dev_auc","dev_auc(0.05)"] + ["tst_auc","tst_auc(0.05)"])
    
    unchanged = 0
    best_dev_score = -1 #here is auc(0.05)
    start_time = 0
    max_epoch = model.args.max_epoch
    for epoch in xrange(max_epoch):
        unchanged += 1
        if unchanged > model.args.max_unchanged: break
        start_time = time.time()
        print "Training Epoch", str(epoch)
        train_loss, dev_eva, test_eva = run_epoch(train, tar_dev, tar_test, model, optimizer_f,optimizer_d)
        dev_auc,dev_auc005 = dev_eva
        test_auc,test_auc005 = test_eva
        
        if  dev_auc005 > best_dev_score:
            best_dev_score = dev_auc005
            unchanged = 0
            result_table.add_row(
                            [ epoch ] +
                            [ "%.2f" % x for x in list(dev_eva) + list(test_eva)])
            if model.args.if_save:
                torch.save(model, os.path.join(model.args.save_model,"model_ep"+str(epoch)+".pkl.gz"))
            say("\r\n\n")

            type((time.time()-start_time)/60.0)
            say(( "Epoch {}\ttrain_loss={:.3f}\tdev_auc={:.3f}\tdev_auc(0.05)={:.3f}\tTime:{:.3f}m\n").format(
                    epoch,
                    train_loss,
                    dev_auc,
                    dev_auc005,
                    (time.time()-start_time)/60.0
            ))
            say("\tp_norm: {}\n".format(
                    model.get_pnorm_stat()
                ))
            say("\n")
            say("{}".format(result_table))
            say("\n")
    
    if not os.path.exists(model.args.save_model):
        os.makedirs(model.args.save_model)
    torch.save(model, os.path.join(model.args.save_model,"model_ep"+str(epoch)+"_best.pkl.gz"))


def run_epoch(train, tar_dev, tar_test, model, optimizer_f, optimizer_d):
    '''
    Run one epoch (one pass of data). Return training loss, dev and test evaluation performance after this epoch. 
    Create new training batches from both source domain and target domain. 
    Number of questions from source and target are equal. 
    Questions from target domain are randomly drawn from target corpus. 
    
    Input:
        train: training data in source domain 
        tar_dev: (pos_pairs,neg_pairs) from dev data in target domain
        tar_test: (pos_pairs,neg_pairs) from test data in target domain 
    Output: 
        avg_train_loss: average training loss over all batches 
        dev_eva: (dev_auc,dev_auc005)
        test_eva: (test_auc,test_auc005)

    '''
    print "\nCreate training batches from source domain..."
    src_batches = create_batches(model.args.src_corpus_ids,train,model.args.batch_size, padding_id=model.args.padding_id, perm=None, pad_left=model.args.pad_left)
    
    print "\nCreate training batches from target domain...\n (Draw same # of questions from target corpus.."
    tar_batches = create_batches_target(src_batches, model.args.tar_corpus_ids, padding_id = model.args.padding_id, pad_left=model.args.pad_left)

    #N=10 ##small num for testing
    N = len(src_batches)
    train_loss_y = 0.0
    train_cost_y = 0.0  
    train_loss_d = 0.0 
    train_loss= 0.0
    #set model to training mode 
    model.train() #set model to train mode  
    for i in xrange(N):
        src_titles,src_bodies,triples=src_batches[i]
        tar_titles,tar_bodies = tar_batches[i]
        titles = np.hstack([src_titles,tar_titles])
        bodies = np.hstack([src_bodies,tar_bodies])
        num_ques = titles.shape[1]
        batch = (titles,bodies,triples)
        
        ##Check if each epoch, newly created batches have been reshuffled 
        ##print "batch title..", batch[0][0]
        
        #True domain labels
        domain_true = torch.zeros(num_ques)
        domain_true[num_ques/2:]=1 
        domain_true = autograd.Variable(domain_true.long())
        
        ##Foward pass
        h_final,domain_pred=model(batch)  #h_final: feature extractor outcome; pred_d: predicted domain (log_softmax)

        ##Compute loss for similarity 
        h_final_src = h_final[:num_ques/2,:]  #first half is from src domain 
        loss_y = max_margin_loss(model.args,h_final_src,triples,model.args.margin)
        cost_y = loss_y + model.get_l2_reg()
        
        ##Compute loss for domain classification
        loss_d = F.nll_loss(domain_pred,domain_true)
        
        ##Total loss 
        loss = loss_y - model.args.lambd * loss_d
        #print 'loss_y,loss_d,loss',loss_y.data[0],loss_d.data[0],loss.data[0]
        
        ##backpropagation
        optimizer_f.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_f.step()
        optimizer_d.step()
        
        train_loss_y += loss_y.data
        train_cost_y += cost_y.data
        train_loss_d += loss_d.data
        train_loss += loss.data
        
## Observe if a subset of parameters change in each iteration 
#        print [p for p in model.parameters()][1].data[0:5]
#        print [p for p in model.parameters()][5].data[0:5]
        
        if i%10 == 0: #print after processing every 10 batches 
            say("\r{}/{}".format(i,N))
            print "\n"
            print "loss_y,cost_y (similarity):", (train_loss_y/(i+1))[0], (train_cost_y/(i+1))[0]
            print "loss_d (domain classification):", (train_loss_d/(i+1))[0]
            print "loss:", (train_loss/(i+1))[0]
            print "\n"
             
        ##Evaluate on TARGET dev/test data at the end of this training epoch 
        avg_train_loss = (train_loss/(i+1))[0]
        dev_eva = None
        test_eva = None
        if i == N-1: #last batch of this epoch
            if tar_dev is not None:
                dev_eva = model.evaluate_auc(tar_dev)
            if tar_test is not None: 
                test_eva = model.evaluate_auc(tar_test)
    return avg_train_loss, dev_eva, test_eva



