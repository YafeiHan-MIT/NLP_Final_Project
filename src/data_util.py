#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:08:54 2017

@author: yafei
"""
import sys
import os
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
os.chdir(dirname(dirname(realpath(__file__)))) #u'/Users/yafeihan/Dropbox (MIT)/Courses_MIT/6.864_NLP/NLP_Final_Project'

import gzip
import numpy as np
import torch
import cPickle as pickle
import sys
import os
import numpy as np 
import random 
from torch.autograd import Variable

def say(s, stream=sys.stdout):
    stream.write(s)
    stream.flush()
    
def read_corpus(path):
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
            title = title.strip().lower().split()
            body = body.strip().lower().split()
            raw_corpus[id]=(title,body)
    say("{} empty title records are ignored.\n".format(empty_cnt))
    return raw_corpus


def read_corpus_get_unique(path):
    '''
    Return a raw corpus dictionary: {id:(title,body)} and set of unique words (Set[string]).

    Input: path to the corpus text file
            text format: id, title and body sep by '\t'

    Output: {id:str: (title:list, body:list)}, Set[string]

    '''
    raw_corpus = {}
    empty_cnt = 0
    vocab = set()
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as f:
        for line in f:
            id, title, body = line.split('\t')
            if len(title) == 0:
                print "empty title: id=", id
                empty_cnt += 1
                continue
            title = title.strip().lower().split()
            body = body.strip().lower().split()
            unique_words = set(title + body)
            vocab.update(unique_words)
            raw_corpus[id] = (title, body)
    say("{} empty title records are ignored.\n".format(empty_cnt))
    return raw_corpus, vocab
   
def words_to_indices(word_list, word_to_indx,if_filter=True):
    '''
    Convert a list of words to a list of word indices 
    Input: 
        word_list: [w0,w1,..wn] 
        word_to_indx: {word:indx}
    Output: 
        array of word indices [idx0,idx1,...idxn]
    '''
    nil_indx = 0  #unknown word index 
    indices = [word_to_indx[w] if w in word_to_indx else nil_indx for w in word_list]
    if if_filter:
        not_oov = lambda x: x!= nil_indx   
        indices = filter(not_oov, indices)
    return np.array(indices)
         
def getEmbeddingTable(embedding_path, vocab=None):
    '''
    Input: 
        a fixed embedding lookup file: word, embedding vector
        vocab: (set) all unique words in training corpus (used to reduce embedding table size)
    Output: 
        embeddings: (vocabSize+1) by embed_dim array
                    embeddings[0,:] = [0,...0] reserved for words not in the embedding vocabulary!
                    embeddings[indx,:] is word indx's embedding vector
        word_to_indx: {word: indx}
    '''
    embeddings = []
    word_to_indx = {}
    indx = 0
    with gzip.open(embedding_path) as file:
        for l in file:
            word, emb = l.split()[0], l.split()[1:]
            if vocab is not None and word not in vocab:
                continue

            vector = [float(x) for x in emb ]
            if indx == 0: # reserve word index 0 for words not in embedding dictionary
                embeddings.append(np.zeros(len(vector)))
                indx += 1
            embeddings.append(vector)
            word_to_indx[word] = indx
            indx += 1
        embeddings = np.array(embeddings, dtype=np.float32)
    return embeddings, word_to_indx

def map_corpus(raw_corpus, embeddings, word_to_indx, max_len=100):
    '''
    Map raw_corpus to ids_corpus (with words replaced by word indices)
    
    Input: 
        raw_corpus:
        embeddings:a lookup array, (vocab_size+1) by embed_dim
    Return: 
        ids_corpus ={id: (title_indices, body_indices)}
            title_indices: 1-d array
            body_indices: 1-d array of max_len
    '''
    ids_corpus = {}
    for id, (title,body) in raw_corpus.iteritems():   
        ids_corpus[id] = (words_to_indices(title, word_to_indx),words_to_indices(body, word_to_indx)[:max_len])
    return ids_corpus

def read_annotations(path, num_neg=20):
    '''
    Return a list of queries with its postive and negative companions: 
        
    Input: text file with pid pos neg separated by '/t'
        pid: source question id
        pos: relevant question ids 
        neg: irrelevant question ids
    Output:[(pid, [qids], [qlabels])]
        pid: source question id
        qids: list of question ids to compare with  
        qlabels: list of question labels (1: pos, 0: neg) 
    
    Example: 
        Input: p  a b  x y z   
                p: source question, a b: pos questions; x y z: neg questions  
        return [(p,[a,b,x,y,z],[1,1,0,0,0])]
    '''
    lst = []
    with open(path) as fin:
        for line in fin:
            pid,pos,neg = line.strip().split('\t')[:3]
            pos = pos.split() #list of pos
            neg = neg.split() #list of neg
            if len(pos) == 0: continue
            if num_neg <> -1:  #num of neg is specified 
                random.shuffle(neg) #shuffle all the negative examples
                neg = neg[:num_neg] #only use num_neg negative examples  
            qids = []
            qlabels = []
            seen = set() #avoid repetition 
            for q in neg:
                if q not in seen:
                    qids.append(q)
                    qlabels.append(0 if q not in pos else 1)
                    seen.add(q)
            for q in pos:
                if q not in seen:
                    qids.append(q)
                    qlabels.append(1)
                    seen.add(q)
            lst.append((pid, qids, qlabels))
    return lst

def create_batches(ids_corpus, data, batch_size, padding_id=0, perm=None, pad_left=False):
    '''
    Input: 
        ids_corpus: question content dictionary {pid:(title,body)}   
            title is a list of word indices; body is a list of word indices 
        data: source questions and their companions' label [(pid:string, qids:list, qlabels:list)]
        batch_size: number of source questions included in a batch 
        
    Output:  
        a list of batches, each batch consists of (titles_padded, bodies_padded, triples_padded)
        [(titles_padded, bodies_padded, triples_padded), ...]
            titles_padded: seq_len_title * num_que_in_batch
            bodies_padded: seq_len_body * num_que_in_batch
            triples_padded: num_triples * (1+candidate_size)
            !!Note: question ids are new ids for each batch.             
    '''
    ##Every time creating batches, reshuffle data first 
    if perm is None: #no perm order is given, shuffle data (0,...N-1) => perm  
        perm = range(len(data)) 
        random.shuffle(perm) #a shuffled indices ranging in (0,...,N-1) 

    N = len(data) # num of observations 
    ##For current batch 
    cnt = 0 
    pid2id = {} #map pid -> serial number acc to current reshuffled order 
    titles = []
    bodies = []
    triples = []
    
    ##Record all batches 
    batches = []
    for u in xrange(N): #loop over each observation in the reshuffled order 
        i = perm[u] # get the original index in data 
        pid, qids, qlabels = data[i] 
        if pid not in ids_corpus: continue 
        cnt += 1 #count how many source questions have been processed, in order to check one batch is filled 
        
        ##Build pid2id dictionary: question id => new id in one batch
        for id in [pid] + qids: #loop over all questions including the source question id. 
            if id not in pid2id: 
                if id not in ids_corpus: continue #skip those question not in corpus 
                pid2id[id] = len(titles) #map question id to a new id in this new batch 
                t, b = ids_corpus[id] #get title and body word indices 
                titles.append(t)  #add title to titles 
                bodies.append(b)  #append body to bodies 
        
        ##Update pid, pos and neg to new id in the current batch 
        pid = pid2id[pid]   #pid's new id 
        pos = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 1 and q in pid2id ] #get pos set new id list 
        neg = [ pid2id[q] for q, l in zip(qids, qlabels) if l == 0 and q in pid2id ] #get neg set new id list 
        triples += [ [pid,x]+neg for x in pos ] 
        #if there are multple pos, for each pos, create a [pid, pos, negs]
        #triples is a list of list, each list is [pid,pos_id(len=1),neg_ids(len=20)]

        if cnt == batch_size or u == N-1: #reach batch_size(num of source questions) or end of data
            titles_padded, bodies_padded = create_one_batch(titles, bodies, padding_id, pad_left)
            triples_padded = create_hinge_batch(triples)
            batches.append((titles_padded, bodies_padded, triples_padded))
            ##Clear these for the next batch 
            titles = []
            bodies = []
            triples = []
            pid2id = {}
            cnt = 0
    return batches 

def create_one_batch(titles, bodies, padding_id=0, pad_left=False, max_title_len = None, max_body_len = None):
    '''
    Creat one padded batch for titles and bodies
    one batch here is a pool of questions, where number of questions = |titles| =|bodies|
    
    Input: 
        titles: list of 1-D array of title word indices
        bodies: list of 1-D array of body word indices
        padding_id: constant value for padding  
        pad_left: True when padding to the left 
    
    Output:
        titlesPadded: array, max_title_len by num of questions in the batch
        bodiesPadded: array, max_body_len by num of questions in the batch 
    '''
    if max_title_len == None:
        max_title_len = max([len(x) for x in titles])
    if max_body_len == None:
        max_body_len = max([len(x) for x in bodies])
    if pad_left:
        list_titles_padded = [np.pad(x,(max_title_len-len(x),0),mode = "constant",constant_values = padding_id) for x in titles]
        list_bodies_padded = [np.pad(x,(max_body_len-len(x),0),mode = "constant",constant_values = padding_id) for x in bodies]
    else:
        list_titles_padded = [np.pad(x,(0,max_title_len-len(x)),mode = "constant",constant_values = padding_id) for x in titles]
        list_bodies_padded = [np.pad(x,(0,max_body_len-len(x)),mode = "constant",constant_values = padding_id) for x in bodies]
    
    # Stack 1-D arrays in a list by column to a 2-D array.
    
    titlesPadded = np.column_stack(list_titles_padded) # max_title_len by num of questions in the batch
    bodiesPadded = np.column_stack(list_bodies_padded) # max_body_len by num of questions in the batch
    return titlesPadded, bodiesPadded

def create_hinge_batch(triples):
    '''
    Pad triples on the right with the edge values 
    Input: 
        triples: a list of list. Each inside list=[pid, pos_id, neg_ids]. pos_id is a SINGLE pos observation. 
    Output:
        triples_padded: array, number of triples * max length of [pid, pos_id, neg_ids]
    '''
    max_len = max(len(x) for x in triples) #max length of [pid, pos_id, neg_ids] list 
    triples_padded = np.vstack([ np.pad(x,(0,max_len-len(x)),'edge')
                        for x in triples ])
    return triples_padded   

def create_eval_batches(ids_corpus, data, padding_id, pad_left):
    '''
    Input: 
        ids_corpus: question content dictionary {pid:(title,body)}   
            title is a list of word indices; body is a list of word indices 
        data: source questions and their companions' label [(pid:string, qids:list, qlabels:list)]
    Output:
        [(titles,bodies,labels),...]
        - titles: seq_len_title, num_ques(1 source query + 20 candidate queries)
        - bodies: seq_len_body, num_ques(21)
        - labels: 1-d array of length = 20 
    '''
    lst = []
    for pid, qids, qlabels in data: #for each query, create one batch 
        titles = []
        bodies = []
        for id in [pid]+qids:
            t,b = ids_corpus[id] #title word list, body word list 
            titles.append(t)
            bodies.append(b)
        titles_padded, bodies_padded = create_one_batch(titles, bodies, padding_id, pad_left)
        lst.append((titles_padded, bodies_padded, np.array(qlabels, dtype="int32")))
    return lst


def create_batches_target(src_batches, tar_corpus_ids, padding_id, pad_left):
    '''
    Input:
        src_batches: source domain training batches
        tar_corpus_ids:target domain corpus (using word indices)
            
    Output:
        target batches: [(titlesPadded,bodiesPadded),(titlesPadded,bodiesPadded),...]       
    '''
    
    tar_batches=[]
    for i in range(len(src_batches)):
        ##max title length and body length should match source batch 
        max_title_len, num_ques=src_batches[i][0].shape
        max_body_len, num_ques=src_batches[i][1].shape
        
        #randomly draw same number of questions from target domain corpus 
        sel_tar_ids=np.random.choice(tar_corpus_ids.keys(),num_ques) 
        titles = [tar_corpus_ids[key][0] for key in sel_tar_ids] #titles of selected target domain questions
        bodies = [tar_corpus_ids[key][1] for key in sel_tar_ids] #bodies of selected target domain questions 
        
        titles_trunc = [t[:max_title_len] for t in titles]
        bodies_trunc = [b[:max_body_len] for b in bodies]
        
        titlesPadded,bodiesPadded = \
        create_one_batch(titles_trunc, bodies_trunc, padding_id=padding_id, pad_left=pad_left, max_title_len = max_title_len, max_body_len=max_body_len)   
        tar_batches.append((titlesPadded,bodiesPadded))
    return tar_batches

def create_eval_batches_target(tar_corpus_ids, pairs, pairs_per_batch, padding_id, pad_left):
    '''
    Create evaluation data out of TARGET domain dev/test data 
    pairs: list of pos pairs OR neg pairs from target domain dev/test data 
    pairs_per_batch: number of pairs to include in each evaluation batch
    '''
    i = 0
    eval_batches = []
    while i<len(pairs):
        pairs_indx = []
        titles = []
        bodies = []
        seen = set()
        unique_ques = []
        for pair in pairs[i:(i+pairs_per_batch)]:
            if pair[0] not in seen:
                unique_ques.append(pair[0])
                seen.add(pair[0])
            if pair[1] not in seen:
                unique_ques.append(pair[1])
                seen.add(pair[1])
        qid_to_indx = {}
        for q in unique_ques:
            titles.append(tar_corpus_ids[q][0])
            bodies.append(tar_corpus_ids[q][1])
            qid_to_indx[q]=len(titles)-1
        for pair in pairs[i:(i+pairs_per_batch)]:
            pairs_indx.append([qid_to_indx[pair[0]],qid_to_indx[pair[1]]])

        titlesPadded,bodiesPadded = create_one_batch(titles, bodies, padding_id, pad_left, max_title_len = None, max_body_len = None)
        eval_batches.append((titlesPadded, bodiesPadded, pairs_indx))
        #reset for a new batch 
        i=i+pairs_per_batch
        pairs_indx = []
        titles = []
        bodies = []
        seen = set()
        unique_ques = []
    return eval_batches

def read_corpus_Android(path):
    '''
    Input: path to the corpus text file
            text format: id, title and body sep by '\t'
    
    Output:
        raw_corpus: {id:str: (title:list, body:list)}
        text_list: a list of raw text strings that combines title and body text  
        id_to_index: {original id (str): index(int) in the text_list}
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
    
def read_annotations_pairs(path):
    '''
    Read target domain annotations 
    '''
            
    f=open(path)
    pairs = []
    for line in f:
        pairs.append(line.split())
    return pairs
    
def read_annotations_target(path_pos,path_neg):
    '''
    Read target domain annotations (dev/test)
    '''
    pos_pairs=read_annotations_pairs(path_pos)
    neg_pairs=read_annotations_pairs(path_neg)
    return (pos_pairs,neg_pairs)


##Testing code             
#    f=open(path_pos)
#    pos_pairs = []
#    for line in f:
#        pos_pairs.append(line.split())
#    f=open(path_neg)
#    neg_pairs = []
#    for line in f:
#        neg_pairs.append(line.split())
#    
#    companions=dict()
#    labels = dict()
#    for item in neg_pairs:
#        if item[0] not in companions:
#            companions[item[0]]=[item[1]]
#            labels[item[0]]=[0]
#        else:
#            companions[item[0]].append(item[1])
#            labels[item[0]].append(0)
#    for item in pos_pairs:
#        if item[0] not in companions:
#            companions[item[0]]=[item[1]]
#            labels[item[0]]=[1]
#        else:
#            companions[item[0]].append(item[1])
#            labels[item[0]].append(1)
#    lst=[]
#    for key in companions:
#        lst.append((key,companions[key],labels[key])) 
#    return lst

###read external embedding table 
#embedding_path='data/vector/vectors_pruned.200.txt.gz'
#embeddings, word_to_indx = getEmbeddingTable(embedding_path)
#print "vocab size:", embeddings.shape[0]-1
#print "embed dim:", embeddings.shape[1]
#
###read raw corpus data 
#corpus_path='data/text_tokenized.txt.gz'
#raw_corpus=read_corpus(corpus_path)
#print "corpus size:", len(raw_corpus)
#
###convert raw_corpus to ids_corpus
#ids_corpus = map_corpus(raw_corpus, embeddings, max_len=100)
#
####read annotation data 
#anoPath = 'data/train_random.txt'
#train = read_annotations(anoPath, num_neg=20)
#train_batches = create_batches(ids_corpus, train, args.batch_size, padding_id=0, perm=None, pad_left=False)
#print "num of training queries:", len(train)
#
##Create dev batches 
#dev_path = 'data/dev.txt'
#dev = read_annotations(dev_path, num_neg=20)
#dev_batches = create_eval_batches(ids_corpus, dev, padding_id=0, pad_left=False)
#print "number of dev queries:", len(dev) ##189
#
###Create test batches 
#test_path = 'data/test.txt'
#test = read_annotations(test_path, num_neg=20)
#test_batches = create_eval_batches(ids_corpus, test, padding_id=0, pad_left=False)
#print "number of test queries:", len(test) ##186

####create batches out of training data 
#batches = create_batches(ids_corpus, train, 40, padding_id=0, perm=None, pad_left=False)
#print "number of batches", len(batches)
#for b in range(0,7):
#    print "batch:", b
#    batch=batches[b]
#    titles,bodies,triples = batch
#    print "(max_seq_len_title,num_que_in_batch)", titles.shape #(max_seq_len_title,num_que_in_batch)
#    print "(max_seq_len_body,num_que_in_batch)", bodies.shape #(max_seq_len_body,num_que_in_batch)




                                      