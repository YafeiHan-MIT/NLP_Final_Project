#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 09:35:52 2017

@author: yafei
"""

import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
os.chdir(dirname(dirname(realpath(__file__)))) #u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

import gzip
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.evaluation import Evaluation 
from src.data_util import *
from src.data_util_Android import *
from src.meter import *

def cos_similarity(tfidf,id1,id2,id_to_index):
    '''
    id1,id2: corpus text id (string)
    tfidf: corpus size * vocab size (array)
    
    '''
    ind1=id_to_index[id1]
    ind2=id_to_index[id2]
    return np.dot(tfidf[ind1],tfidf[ind2])

##Obtain tfidf for entire Android corpus
corpus_path = 'data/Android/corpus.tsv.gz'
raw_corpus,text_list,id_to_index = read_corpus_Android(corpus_path)
vec = TfidfVectorizer(stop_words='english')
tfidf = vec.fit_transform(text_list).toarray()
vocab = vec.get_feature_names()
print 'size of vocab:', len(vocab)

print "\nRead annotations from target domain (dev/test)"
tar_dev = read_annotations_target('data/Android/dev.pos.txt',
                                  'data/Android/dev.neg.txt')  # a tuple of 2: (pos_pairs,neg_pairs)
tar_test = read_annotations_target('data/Android/test.pos.txt',
                                       'data/Android/test.neg.txt')

    
tar_pos,tar_neg = tar_dev
AUC = AUCMeter()
pos_scores = []
for pair in tar_pos:
    pos_scores.append(cos_similarity(tfidf,pair[0],pair[1],id_to_index))   
AUC.add(np.array(pos_scores), np.ones(len(pos_scores)))

neg_scores = []
for pair in tar_neg:
     neg_scores.append(cos_similarity(tfidf,pair[0],pair[1],id_to_index))   
AUC.add(np.array(neg_scores), np.zeros(len(neg_scores)))
print 'dev:', AUC.value(max_fpr=1.0), AUC.value(max_fpr=0.05)


tar_pos,tar_neg = tar_test
AUC = AUCMeter()
pos_scores = []
for pair in tar_pos:
    pos_scores.append(cos_similarity(tfidf,pair[0],pair[1],id_to_index))   
AUC.add(np.array(pos_scores), np.ones(len(pos_scores)))

neg_scores = []
for pair in tar_neg:
     neg_scores.append(cos_similarity(tfidf,pair[0],pair[1],id_to_index))   
AUC.add(np.array(neg_scores), np.zeros(len(neg_scores)))
print 'test:', AUC.value(max_fpr=1.0), AUC.value(max_fpr=0.05)




#def rank(pos_pairs,neg_pairs,tfidf,id_to_index):
#    ranked_labels = {}
#    ranked_scores = {}
#    for p in pos_pairs:
#        #print 'p=',p
#        scores=[]
#        q_ids=pos_pairs[p]+neg_pairs[p]
#        #print q_ids
#        labels = np.array([1.0]*len(pos_pairs[p])+[0.0]*len(neg_pairs[p]))  ##e.g. array([1,1,0,0...0])
#        #print 'labels:',labels
#        for q in q_ids:
#            scores.append(cos_similarity(tfidf,p,q,id_to_index))
#        sorted_ind= np.argsort(-np.array(scores)) #descending order sorting scores, return indices
#        sorted_scores = np.sort(-np.array(scores))
#        labels_sorted=labels[sorted_ind]
#        ranked_labels[p]=labels_sorted
#        ranked_scores[p]=-sorted_scores
#    return ranked_labels,ranked_scores


#def evaluate(labels_ranked):
#    '''
#    data is a list of golden labels ranked by scores. 
#        e.g. [[1,0],[1,1,1,0],[0,1,0]]
#    '''
#    e=Evaluation(labels_ranked)
#    return e.MAP(),e.MRR(), e.Precision_at_R(1), e.Precision_at_R(5)

#def evaluate_by_cos_similarity(path_pos_pairs,path_neg_pairs,tfidf,id_to_index):
#    pos_pairs=read_labeled_pairs(path_pos_pairs)    
#    neg_pairs=read_labeled_pairs(path_neg_pairs)
#    ranked_labels,ranked_scores=rank(pos_pairs,neg_pairs,tfidf,id_to_index)           
#    ranked = [list(ranked_labels[key]) for key in ranked_labels]
#    e=Evaluation(ranked)
#    say("{:.3f},{:.3f},{:.3f},{:.3f}".format(e.MAP(),e.MRR(), e.Precision_at_R(1), e.Precision_at_R(5)))
#    return e.MAP(),e.MRR(), e.Precision_at_R(1), e.Precision_at_R(5)
#
#def evaluate_auc_cos_similarity(path_pos_pairs,path_neg_pairs,tfidf,id_to_index):
#    pos_pairs=read_labeled_pairs(path_pos_pairs)    
#    neg_pairs=read_labeled_pairs(path_neg_pairs)
#    #ranked_labels,ranked_scores=rank(pos_pairs,neg_pairs,tfidf,id_to_index)           
#    #ranked = [list(ranked_labels[key]) for key in ranked_labels]
#    #e=Evaluation(ranked)
#    #say("{:.3f},{:.3f},{:.3f},{:.3f}".format(e.MAP(),e.MRR(), e.Precision_at_R(1), e.Precision_at_R(5)))
#    return pos_pairs, neg_pairs
#

#pos_pairs, neg_pairs = evaluate_auc_cos_similarity('data/Android/dev.pos.txt','data/Android/dev.neg.txt',tfidf,id_to_index)
##Evaluate performance based on ranking with cosine similarity
#print '\ndev set:'
#print 'MAP, MRR, P@1, P@5\n'
#ev_dev = evaluate_by_cos_similarity('data/Android/dev.pos.txt','data/Android/dev.neg.txt',tfidf,id_to_index)
#
#print '\ntest set:'
#print 'MAP, MRR, P@1, P@5\n'
#ev_test = evaluate_by_cos_similarity('data/Android/test.pos.txt','data/Android/test.neg.txt',tfidf,id_to_index)

