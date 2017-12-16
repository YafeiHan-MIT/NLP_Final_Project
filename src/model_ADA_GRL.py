# import path related modules 
import os, sys, math, nltk
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__)))) ##add project path to system path list
os.chdir(dirname(dirname(realpath(__file__)))) 

import torch.nn as nn
from torch import autograd
import torch.nn.functional as F

from src.init_util import get_activation_by_name
from src.data_util import *
from src.meter import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

    def evaluate_auc(self, data):
        '''
        Evaluate on target domain data  (dev/test)
        data:  target domain data  (dev/test)

        Return: AUC and AUC(0.05)
        '''
        pos_pairs, neg_pairs = data
        AUC = AUCMeter()
        # print 'build eval batche?s from tar_dev (pos pairs)'
        pos_batches = create_eval_batches_target(self.args.tar_corpus_ids, pos_pairs, pairs_per_batch=2000,
                                                 padding_id=self.args.padding_id, pad_left=self.args.pad_left)
        # print 'size of pos_batches:', len(pos_batches)
        count = 1
        for pos_batch in pos_batches:
            # print str(count), 'eval batches (positive pairs) processed'
            count += 1
            pos_titles, pos_bodies, pos_pairs_indx = pos_batch
            pos_labels = np.ones(len(pos_pairs_indx))
            pos_h_final, o = self.forward(pos_batch)
            if self.args.cuda:
                pos_h_final = pos_h_final.data.cpu().numpy()
            else:
                pos_h_final = pos_h_final.data.numpy()
            pos_scores = []
            for pair in pos_pairs_indx:  # compute score for each pair
                pos_scores.append(np.dot(pos_h_final[pair[0]], pos_h_final[pair[1]]))
            AUC.add(np.array(pos_scores), pos_labels)

        # print 'build eval batches from tar_dev (neg pairs)'
        neg_batches = create_eval_batches_target(self.args.tar_corpus_ids, neg_pairs, pairs_per_batch=2000,
                                                 padding_id=self.args.padding_id, pad_left=self.args.pad_left)
        # print 'size of neg_batches:', len(neg_batches)

        count = 1
        for neg_batch in neg_batches:
            # print str(count), 'eval batches (negative pairs) processed'
            count += 1
            neg_titles, neg_bodies, neg_pairs_indx = neg_batch
            neg_labels = np.zeros(len(neg_pairs_indx))
            neg_h_final, o = self.forward(neg_batch)
            if self.args.cuda:
                neg_h_final = neg_h_final.data.cpu().numpy()
            else:
                neg_h_final = neg_h_final.data.numpy()
            neg_scores = []
            for pair in neg_pairs_indx:  # compute score for each pair
                neg_scores.append(np.dot(neg_h_final[pair[0]], neg_h_final[pair[1]]))
            AUC.add(np.array(neg_scores), neg_labels)

        return AUC.value(max_fpr=1.0), AUC.value(max_fpr=0.05)

    def get_pnorm_stat(self):
        '''
        get params norms
        '''
        lst_norms = []
        for p in self.parameters():
            lst_norms.append("{:.3f}".format(p.norm(2).data[0]))
        return lst_norms

    def get_l2_reg(self):
        l2_reg = None
        for p in self.parameters():
            if l2_reg is None:
                l2_reg = p.norm(2)
            else:
                l2_reg = l2_reg + p.norm(2)
        l2_reg = l2_reg * self.args.l2_reg
        return l2_reg


class CNN_ADA(Model):
    def __init__(self, args, embeddings):
        super(CNN_ADA, self).__init__(args)

        args.vocab_size, args.embedding_dim = embeddings.shape
        self.args = args
        self.lambd = args.lambd

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, args.padding_id)
        if args.cuda:
            self.embedding = self.embedding.cuda()
        self.embedding.weight.data = torch.from_numpy(embeddings)  # fixed embedding

        filter_width = 3
        self.conv = nn.Conv1d(args.embedding_dim, args.hidden_dim, filter_width)
        if args.average:
            self.pool = nn.AvgPool1d(filter_width)
        else:
            self.pool = nn.MaxPool1d(filter_width)

        self.linear1 = nn.Linear(args.hidden_dim, args.hidden_dim_dc)
        self.linear2 = nn.Linear(args.hidden_dim_dc, 2)  # from hidden to output 2 domain classes
        self.log_softmax = nn.LogSoftmax()

    def forward(self, batch):
        titles, bodies, triples = batch
        titles = Variable(torch.from_numpy(titles).long(), requires_grad=False)
        bodies = Variable(torch.from_numpy(bodies).long(), requires_grad=False)

        if self.args.cuda:
            titles = titles.cuda()
            bodies = bodies.cuda()

        ## embedding layer: word indices => embeddings
        embeds_titles = self.embedding(titles)  # seq_len_title * num_ques * embed_dim
        embeds_bodies = self.embedding(bodies)  # seq_len_body * num_ques * embed_dim

        # turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len)
        xt = embeds_titles.transpose(0, 1).transpose(1, 2)
        xb = embeds_bodies.transpose(0, 1).transpose(1, 2)

        xt = F.tanh(self.conv(xt))
        xb = F.tanh(self.conv(xb))

        xt = F.tanh(self.pool(xt))
        xb = F.tanh(self.pool(xb))

        xt = normalize_3d(xt)
        xb = normalize_3d(xb)

        h_final = normalize_2d(xt.mean(2) + xb.mean(2))
        h = gradient_reverse(h_final, self.lambd)  #####Apply GRL before domain classification
        h = self.linear1(h)
        h = F.relu(h)
        o = self.linear2(h)
        o = self.log_softmax(o)
        return h_final, o


class LSTM_ADA(Model):
    '''
    LSTM for learning similarity between questions 
    Adverserial Domain Adaptation 
    
    '''
    def __init__(self, args, embeddings):
        '''
        embeddings: fixed embedding table (2-D array, dim=vocab_size * embedding_dim: 100407x200)
        '''
        super(LSTM_ADA, self).__init__(args)
        args.vocab_size, args.embedding_dim = embeddings.shape
        self.args = args # pass on args together to the model
        self.lambd = args.lambd

        ##Layers
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, args.padding_id)
        if args.cuda:
            self.embedding = self.embedding.cuda()
        self.embedding.weight.data = torch.from_numpy(embeddings) #fixed embedding 
        self.lstm = nn.LSTM(
                            input_size = args.embedding_dim, 
                            hidden_size = args.hidden_dim, 
                            num_layers = args.hidden_layers, 
                            dropout=args.dropout)
        self.activation = get_activation_by_name('tanh')  ##can choose other activation function specified in args 
        self.linear1 = nn.Linear(args.hidden_dim,args.hidden_dim_dc)
        self.linear2 = nn.Linear(args.hidden_dim_dc,2) #from hidden to output 2 domain classes
        self.log_softmax = nn.LogSoftmax()
            
    def init_hidden(self,num_ques):
        '''
        Input:
            num_ques: number of unique questions (source query & candidate queries) in the current batch. 
            (NOTE:  diff from batch_size := num of source questions)             
        Return (h_0, c_0): hidden and cell state at position t=0
            h_0 (num_layers * num_directions=1, batch, hidden_dim): tensor containing the initial hidden state for each element in the batch.
            c_0 (num_layers * num_directions=1, batch, hidden_dim): tensor containing the initial cell state for each element in the batch.
        '''
        t = autograd.Variable(torch.zeros(self.args.hidden_layers, num_ques, self.args.hidden_dim),requires_grad = True)
        b = autograd.Variable(torch.zeros(self.args.hidden_layers, num_ques, self.args.hidden_dim),requires_grad = True)
        if self.args.cuda:
            t = t.cuda()
            b = b.cuda()
        return (t, b)

    def forward(self, batch):
        '''
        Pass one batch
        Input:
            batch: one batch is a tuple (titles:2-d array, bodies:2-d array, triples:2-d array)
                titles: padded title word idx list for all titles in the batch  (seq_len_title * num_ques)
                bodies: padded body word idx list for all bodies in the batch  ((seq_len_body * num_ques))
                triples: each row is a source query and its candidate queries (1 pos,20 neg )
                        batch_size * 21
        Output:
            self.h_t: Variable, seq_len_title * num_ques * hidden_dim 
            self.h_b: Variable, seq_len_body * num_ques * hidden_dim
            self.h_final: Tensor, num_ques * hidden_dim
        '''
        titles,bodies,triples = batch 
        seq_len_t, num_ques= titles.shape
        seq_len_b, num_ques= bodies.shape
        self.hcn_t = self.init_hidden(num_ques) #(h_0, c_0) for titles' initial hidden states
        self.hcn_b = self.init_hidden(num_ques) #(h_0, c_0) for bodies' initial hidden states

        titles = Variable(torch.from_numpy(titles).long(), requires_grad=False)
        bodies = Variable(torch.from_numpy(bodies).long(), requires_grad=False)

        if self.args.cuda:
            titles = titles.cuda()
            bodies = bodies.cuda()
        
        ## embedding layer: word indices => embeddings 
        embeds_titles = self.embedding(titles) #seq_len_title * num_ques * embed_dim
        embeds_bodies = self.embedding(bodies) #seq_len_body * num_ques * embed_dim
        
        ## lstm layer: word embedding (200) & h_(t-1) (hidden_dim) => h_t (hidden_dim)
        h_t, self.hcn_t = self.lstm(embeds_titles, self.hcn_t)
        h_b, self.hcn_b = self.lstm(embeds_bodies, self.hcn_b)
        
        ## activation function 
        h_t = self.activation(h_t) #seq_len * num_ques * hidden_dim
        h_b = self.activation(h_b) #seq_len * num_ques * hidden_dim
        
        #if args.normalize:
        h_t = normalize_3d(h_t)
        h_b = normalize_3d(h_b)
        
        self.h_t = h_t #self.h_t: seq_len * num_ques * hidden_dim
        self.h_b = h_b #self.h_b: seq_len * num_ques * hidden_dim
        
        if self.args.average: # Average over sequence length, ignoring paddings
            h_t_final = self.average_without_padding(h_t, titles) #h_t: num_ques * hidden_dim
            h_b_final = self.average_without_padding(h_b, bodies) #h_b: num_ques * hidden_dim
        else:  #last pooling
            h_t_final = self.last_without_padding(h_t,titles)
            h_b_final = self.last_without_padding(h_b,bodies)
        #Pool title and body hidden tensor together 
        h_final = (h_t_final+h_b_final)*0.5 # num_ques * hidden_dim
        #h_final = apply_dropout(h_final, dropout) ???
        h_final = normalize_2d(h_final) ##normalize along hidden_dim, hidden representation of a question has norm = 1
        
        # first half: source domain questions  
        # second half: target domain questions  
        
        h = gradient_reverse(h_final, self.lambd) #####Apply GRL before domain classification
        
        h = self.linear1(h)
        h = F.relu(h)
        o = self.linear2(h)
        o = self.log_softmax(o)
        return h_final,o
        
    def average_without_padding(self, x, ids,eps=1e-8):
        '''
        average hidden output over seq length ignoring padding 
        
        Input: 
            x: Variable that contains hidden layer output tensor; size = seq_len * num_ques * hidden_dim
        Output: 
            avg: num_ques * hidden_dim
        '''
        mask = (ids<>self.args.padding_id) * 1
        seq_len, num_ques = mask.size()
        mask_tensor = mask.data.float().view((seq_len, num_ques,-1)) #mask_tensor: seq_len * batch * 1
        mask_tensor = mask_tensor.expand((seq_len, num_ques,self.args.hidden_dim)) #repeat the last dim to match hidden layer dimension
        mask_variable = Variable(mask_tensor,requires_grad = True)

        if self.args.cuda:
            mask_variable = mask_variable.cuda()

        avg = (x*mask_variable).sum(dim=0)/(mask_variable.sum(dim=0)+eps)
        return avg
    
    def last_without_padding(self,x,ids):
        '''
        Last hidden output of a sequence ignoring padding 
        
        Input: 
            x: Variable that contains hidden layer output tensor; size = seq_len * num_ques * hidden_dim
            ids: actual titles or bodies word indices padded with 0 on the right(seq_len * num_ques) 
        Output: 
            last hidden output: num_ques * hidden_dim       
        '''
        seq_len, num_ques = ids.data.size()
        mask_tensor = torch.zeros(seq_len, num_ques,self.args.hidden_dim)
        #find seq last position
        last_ind = ((ids.data<>0).float().sum(dim=0)-1.0).long() #1D tensor: index of the seq last position
        for i in range(num_ques):
            mask_tensor[last_ind[i],i,:]=1 #put 1 to the last position of the sequence in mask. 
        mask_variable = Variable(mask_tensor,requires_grad = True) #seq_len, num_ques, hidden_dim
        
        if self.args.cuda:
            mask_variable = mask_variable.cuda() 
        return (x*mask_variable).sum(dim=0) ##num_ques by hidden_dim
    

def max_margin_loss(args,h_final,triples,margin):
    '''
    Post process h_final: Compute average max margin loss for a batch 
    '''
    hidden_dim = h_final.size(1)

    idps_tensor = Variable(torch.from_numpy(triples.ravel()).long())
    if args.cuda:
        idps_tensor = idps_tensor.cuda()
    
    queSet_vectors = torch.index_select(h_final, 0, idps_tensor) #flatten triples question indices to a 1-D Tensor of len = source queries *22
    queSet_vectors = queSet_vectors.view(triples.shape[0],triples.shape[1],hidden_dim) #num of query * 22 * hidden_dim
    
    # num of query * hidden_dim
    src_vecs = queSet_vectors[:,0,:] #source query * hidden_dim (source query)
    pos_vecs = queSet_vectors[:,1,:]  #source query * hidden_dim  (1 pos sample)
    neg_vecs = queSet_vectors[:,2:,:] #source query * 20 * hidden_dim   (20 negative samples )
    
    pos_scores = (src_vecs * pos_vecs).sum(dim = 1) # 1-d Tensor: num queries 
    
    #add 1 dimemnsion, and repeat to match neg_vecs shape 
    src_vecs_repeat = src_vecs.double().view(src_vecs.size()[0],-1,src_vecs.size()[1]).expand(neg_vecs.size()).float()
    neg_scores = (src_vecs_repeat * neg_vecs).sum(dim = 2) #cosine similarity: sum product over hidden dimension. source query * 20
    neg_scores_max,index_max = neg_scores.max(dim=1) # max over all 20 negative samples  # 1-d Tensor: num source queries 
    diff = neg_scores_max - pos_scores + margin   #1-d tensor  length = num of source queries    
    loss = ((diff>0).float()*diff).mean() #average loss over all source queries in a batch 
    return loss


def cosSim(h_final):
    '''
        Post process h_final for dev or test: Compute the cosine similarities
        first row in batch is source query, the rest are candidate questions
    '''
    hidden_dim = h_final.size(1)
    source = h_final[0] #first row in h_final is the source query's hidden layer output  
    candidates = h_final[1:] #2nd row beyond in h_final are the candidate query's hidden layer output  
    source = source.view(-1,hidden_dim).expand(candidates.size()) 
    cosSim = (source * candidates).sum(dim=1) 
    return cosSim

def normalize_3d(x,eps=1e-8):
        '''
        Normalize a Variable containing a 3d tensor on axis 2
        Input: Variable 
            x: seq_len * num_ques * hidden_dim
        Output: Variable 
            a normalized x (normalized on the last dim)
        '''
        l2 = x.norm(p=2,dim=2).view(x.data.shape[0],x.data.shape[1],1) 
        #make sure l2 dim = x dim = seq_len * num_ques * hidden_dim
        return x/(l2+eps)

def normalize_2d(x, eps=1e-8):
    # x is batch*hidden_dim
    # l2 is batch*1
    l2 = x.norm(2,dim=1)  #l2: 1d tensor of dim = num_ques
    l2 = l2.view(len(l2),-1) #change l2's dimension to: num_ques * 1
    return x/(l2+eps)  


###Alternative implementation: use GRL layer 
class GradReverse(autograd.Function):
    def __init__(self,lambd):
        self.lambd = lambd
   # @staticmethod
    def forward(self, input):
        return input.view_as(input)
    #@staticmethod
    def backward(self, grad_output):
        return (- grad_output * self.lambd)

def gradient_reverse(x,lambd):
    return GradReverse(lambd)(x)
    



 
    