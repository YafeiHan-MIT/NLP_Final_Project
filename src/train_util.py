#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:05:10 2017

@author: yafei
"""

from prettytable import PrettyTable
import time
from src.model_lstm import *


def train_model(ids_corpus, model, train, dev, test):
    '''
    Input: 
        train_data: 
        dev_data: 
        model: a sublass of torch.nn.Module
    '''
    args = model.args
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)

    if args.cuda:
        model = model.cuda()

    result_table = PrettyTable(["Epoch", "dev MAP", "dev MRR", "dev P@1", "dev P@5"] +
                               ["tst MAP", "tst MRR", "tst P@1", "tst P@5"])
    unchanged = 0
    best_dev_MRR = -1
    max_epoch = args.max_epoch
    for epoch in xrange(max_epoch):
        unchanged += 1
        if unchanged > 15: break
        start_time = time.time()
        train_loss, train_cost, dev_eva, test_eva = run_epoch(ids_corpus, train, dev, test, model, optimizer, args)

        dev_MRR = dev_eva[1]
        if dev_MRR > best_dev_MRR:
            best_dev_MRR = dev_eva[1]
            unchanged = 0
            result_table.add_row(
                [epoch] +
                ["%.2f" % x for x in list(dev_eva) + list(test_eva)])
            if args.if_save:
                torch.save(model, os.path.join(args.save_model, "model_ep" + str(epoch) + ".pkl.gz"))
        say("\r\n\n")

        type((time.time() - start_time) / 60.0)
        say(("Epoch {}\tcost={:.3f}\tloss={:.3f}\tdev_MRR={:.2f}\tTime:{:.3f}m\n").format(
            epoch,
            train_cost,
            train_loss,
            dev_MRR,
            (time.time() - start_time) / 60.0
        ))
        say("\tp_norm: {}\n".format(
            model.get_pnorm_stat()
        ))
        say("\n")
        say("{}".format(result_table))
        say("\n")
    torch.save(model, os.path.join(args.save_model, "model_ep" + str(epoch) + "_best.pkl.gz"))


def run_epoch(ids_corpus, train, dev, test, model, optimizer, args):
    '''
    Run one epoch (one pass of data)
    Return average training loss, training cost, dev and test evaluations after this epoch. 
    '''

    #set model to training mode 
    model.train() #set model to train mode 
    train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id=0, perm=None, pad_left=args.pad_left) 

    N = len(train_batches)
    train_loss = 0.0
    train_cost = 0.0
    for i in xrange(N):
        batch = train_batches[i]
        # print "batch title..", batch[0][0]
        h_final = model(batch)
        loss = max_margin_loss(args, h_final, batch, args.margin)
        cost = loss + model.get_l2_reg()

        ##backprop
        optimizer.zero_grad()
        loss.backward()  # back propagation, compute gradient
        optimizer.step()  # update parameters

        train_loss += loss.data
        train_cost += cost.data
        if i % 10 == 0:
            say("\r{}/{}".format(i, N))
        dev_eva = None
        test_eva = None
        if i == N - 1:  # last batch
            if dev is not None:
                dev_eva = model.evaluate(dev)
            if test is not None:
                test_eva = model.evaluate(test)
    return (train_loss / (i + 1))[0], (train_cost / (i + 1))[0], dev_eva, test_eva
