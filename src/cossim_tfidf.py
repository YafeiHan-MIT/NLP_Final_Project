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
import src.evaluation

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()

def read_corpus_Android(path):
    '''
    Return a raw corpus dictionary: {id:(title,body)}.
    
    Input: path to the corpus text file
            text format: id, title and body sep by '\t'
    
    Output: {id:str: (title:list, body:list)}

    '''
    raw_corpus={}
    empty_cnt = 0
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as f:
        for line in f:
            id,title,body = line.split('\t')
            if len(title)==0:
                print "empty title: id=",id
                empty_cnt+=1
                continue
            raw_corpus[id]=(title+' '+body).strip()
    print empty_cnt,"empty title records are ignored.\n"
    id_to_index = dict()
    text_list = []
    i=0
    for key in raw_corpus:
        id_to_index[key]=i
        text_list.append(raw_corpus[key])
        i+=1
    return raw_corpus,text_list,id_to_index

def read_labeled_pairs(path):
    pairs=dict()
    with open(path) as f:
        for line in f:
            pid,qid=line.split()
            if pid not in pairs:
                pairs[pid]=[qid]
            else:
                pairs[pid].append(qid)
    return pairs

def cos_similarity(tfidf,id1,id2,id_to_index):
    '''
    id1,id2: corpus text id (string)
    tfidf: corpus size * vocab size (array)
    
    '''
    ind1=id_to_index[id1]
    ind2=id_to_index[id2]
    return np.dot(tfidf[ind1],tfidf[ind2])
 
def rank(pos_pairs,neg_pairs,tfidf,id_to_index):
    ranked_labels = {}
    ranked_scores = {}
    for p in pos_pairs:
        #print 'p=',p
        scores=[]
        q_ids=pos_pairs[p]+neg_pairs[p]
        #print q_ids
        labels = np.array([1.0]*len(pos_pairs[p])+[0.0]*len(neg_pairs[p]))  ##e.g. array([1,1,0,0...0])
        #print 'labels:',labels
        for q in q_ids:
            scores.append(cos_similarity(tfidf,p,q,id_to_index))
        sorted_ind= np.argsort(-np.array(scores)) #descending order sorting scores, return indices
        sorted_scores = np.sort(-np.array(scores))
        labels_sorted=labels[sorted_ind]
        ranked_labels[p]=labels_sorted
        ranked_scores[p]=-sorted_scores
    return ranked_labels,ranked_scores
        

def evaluate(labels_ranked):
    '''
    data is a list of golden labels ranked by scores. 
        e.g. [[1,0],[1,1,1,0],[0,1,0]]
    '''
    e=src.evaluation.Evaluation(labels_ranked)
    return e.MAP(),e.MRR(), e.Precision_at_R(1), e.Precision_at_R(5)

def evaluate_by_cos_similarity(path_pos_pairs,path_neg_pairs,tfidf,id_to_index):
    pos_pairs=read_labeled_pairs(path_pos_pairs)    
    neg_pairs=read_labeled_pairs(path_neg_pairs)
    ranked_labels,ranked_scores=rank(pos_pairs,neg_pairs,tfidf,id_to_index)           
    ranked = [list(ranked_labels[key]) for key in ranked_labels]
    ev = evaluate(ranked)
    say("{:.3f},{:.3f},{:.3f},{:.3f}".format(ev[0],ev[1],ev[2],ev[3]))
    return ev

##Obtain tfidf for entire Android corpus
corpus_path = 'data/Android/corpus.tsv.gz'
raw_corpus,text_list,id_to_index = read_corpus_Android(corpus_path)
vec = TfidfVectorizer(stop_words='english')
tfidf = vec.fit_transform(text_list).toarray()
vocab = vec.get_feature_names()
print 'size of vocab:', len(vocab)

##Evaluate performance based on ranking with cosine similarity
print 'dev set:'
print 'MAP, MRR, P@1, P@5\n'
ev_dev = evaluate_by_cos_similarity('data/Android/dev.pos.txt','data/Android/dev.neg.txt',tfidf,id_to_index)

print 'test set:'
print 'MAP, MRR, P@1, P@5\n'
ev_test = evaluate_by_cos_similarity('data/Android/test.pos.txt','data/Android/test.neg.txt',tfidf,id_to_index)

