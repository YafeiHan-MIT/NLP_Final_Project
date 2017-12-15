#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:05:10 2017

@author: yafei
"""

from prettytable import PrettyTable
import time
from src.model_lstm_adda import *


def train_source_model(model, train, dev, test):
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
    best_dev = -1
    max_epoch = args.max_epoch
    for epoch in xrange(max_epoch):
        unchanged += 1
        if unchanged > 15: break
        start_time = time.time()
        train_loss, train_cost, dev_eva, test_eva = run_src_train_epoch(train, dev, test, model, optimizer, args)

        dev_MRR = dev_eva[1]
        if dev_MRR > best_dev:
            best_dev = dev_eva[1]
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


def run_src_train_epoch(train, dev, test, model, optimizer, args):
    '''
    Run one epoch (one pass of data)
    Return average training loss, training cost, dev and test evaluations after this epoch. 
    '''
    # set model to training mode
    model.train()  # set model to train mode
    train_batches = create_batches(args.src_corpus_ids, train, args.batch_size, padding_id=0, perm=None, pad_left=args.pad_left)
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


def train_target_model(args, source_model, target_model, discriminator, train, tar_dev, tar_test, optimizer_d,
                       optimizer_m):
    result_table = PrettyTable(["Epoch", "dev_auc", "dev_auc(0.05)"] + ["tst_auc", "tst_auc(0.05)"])

    unchanged = 0
    best_dev_score = -1  # here is auc(0.05)
    start_time = 0
    max_epoch = args.max_epoch
    for epoch in xrange(max_epoch):
        unchanged += 1
        if unchanged > 15: break
        start_time = time.time()
        train_loss_m, train_loss_d, dev_eva, test_eva = run_tar_train_epoch(args, source_model, target_model,
                                                                            discriminator, train, tar_dev, tar_test,
                                                                            optimizer_d, optimizer_m)
        dev_auc, dev_auc005 = dev_eva
        test_auc, test_auc005 = test_eva

        if dev_auc005 > best_dev_score:
            best_dev_score = dev_auc005
            unchanged = 0
            result_table.add_row(
                [epoch] +
                ["%.2f" % x for x in list(dev_eva) + list(test_eva)])
            say("\r\n\n")

            print "Epoch ", epoch
            print "dev auc: %f, dev auc(0.05): %f" % (dev_auc, dev_auc005)
            say("\tp_norm: {}\n".format(
                target_model.get_pnorm_stat()
            ))
            say("\n")
            say("{}".format(result_table))
            say("\n")


def run_tar_train_epoch(args, source_model, target_model, discriminator, train, tar_dev, tar_test, optimizer_d,
                        optimizer_m):
    # set model to train mode
    discriminator.train()
    target_model.train()

    # create batch from source corpus
    src_batches = create_batches(args.src_corpus_ids, train, args.batch_size,
                                 padding_id=args.padding_id, perm=None, pad_left=args.pad_left)
    # create batch from target (same number as src_batches)
    tar_batches = create_batches_target(src_batches, args.tar_corpus_ids, padding_id=args.padding_id,
                                        pad_left=args.pad_left)

    # create multiple tar_batches for training M
    tar_batches_m = [tar_batches]
    for i in range(3):
        tar_batches_m.append(create_batches_target(src_batches, args.tar_corpus_ids, padding_id=args.padding_id,
                                        pad_left=args.pad_left))

    N = len(src_batches)
    train_loss_m = 0.0
    train_loss_d = 0.0

    for i in xrange(N):

        optimizer_d.zero_grad()

        src_titles, src_bodies, triples = src_batches[i]

        # get representation from source and target models
        src_h_final = source_model(src_batches[i])
        tar_h_final = target_model(tar_batches[i])

        num_ques = src_titles.shape[1]

        # train D on source
        src_pred = discriminator(src_h_final)
        src_pred = src_pred.view(src_pred.size(0))
        expected_ones = Variable(torch.ones(num_ques))
        if args.cuda:
            expected_ones = expected_ones.cuda()

        src_loss_d = F.binary_cross_entropy(src_pred, expected_ones)
        src_loss_d.backward(retain_graph=True)

        # train D on target
        tar_pred = discriminator(tar_h_final)
        tar_pred = tar_pred.view(tar_pred.size(0))
        expected_zeros = Variable(torch.zeros(num_ques))
        if args.cuda:
            expected_zeros = expected_zeros.cuda()

        tar_loss_d = F.binary_cross_entropy(tar_pred, expected_zeros)
        tar_loss_d.backward()
        optimizer_d.step()

        # see how well M fooled the discriminator
        for batch in tar_batches_m:
            optimizer_m.zero_grad()
            tar_h_final = target_model(batch[i])
            tar_pred = discriminator(tar_h_final)
            tar_pred = tar_pred.view(tar_pred.size(0))
            loss_m = F.binary_cross_entropy(tar_pred, expected_ones)
            loss_m.backward()
            optimizer_m.step()

            train_loss_m += loss_m.data

        train_loss_d += src_loss_d.data + tar_loss_d.data

        if i % 10 == 0:
            say("\r{}/{}".format(i, N))

        dev_eva = None
        test_eva = None
        if i == N - 1:  # last batch of this epoch
            if tar_dev is not None:
                dev_eva = target_model.evaluate_auc(tar_dev)
            if tar_test is not None:
                test_eva = target_model.evaluate_auc(tar_test)

            print "\n"
            print "loss_m: ", (train_loss_m / (i + 1))[0]
            print "loss_d (domain classification):", (train_loss_d / (i + 1))[0]
            print "\n"
    return train_loss_m, train_loss_d, dev_eva, test_eva
