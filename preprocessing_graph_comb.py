import numpy as np
import scipy.sparse
import math, random, json
import time, sys, subprocess, os
import glob, pickle, json

import torch

from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import EnglishStemmer

from WikiAnalyzerUtils import Check_Entity_In_Doc

# check occurances of candidates in documents
def check(s, wi, c, stemmer):
    return sum([stemmer.stem(s[wi + j].lower()) == stemmer.stem(c_) for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c) or \
           sum([s[wi + j].lower().split('-')[0] == c_ for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c) or \
           sum([s[wi + j].lower().split('-')[-1] == c_ for j, c_ in  enumerate(c) if wi + j < len(s)]) == len(c)

def ind(si, wi, ci, c):
    return [si, wi, wi+len(c), ci]

class whDataset_comb(Dataset):

    '''
    Dataset class to read in scp and label files
    '''
    def __init__(self, json_file, vocab_file, num_sub):

        self.json_file = json_file
        self.vocab_file = vocab_file

        with open(vocab_file, 'rb') as bid:
            self.vocab_dict = pickle.load(bid)

        self.vocab_size = len(self.vocab_dict)

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.max_cand = 79 # maximum number of candidates
        self.max_docs = 63 # maximum number of docs
        self.max_ment = 700
        self.max_sub_ment = num_sub

        self.max_query_len = 0
        self.max_doc_len = 0
        self.max_cand_len = 0

        self.tokenize = TweetTokenizer().tokenize
        self.stemmer = EnglishStemmer()

        self.parse_json()

    def _w2id(self, word):
        widx = self.vocab_dict.get(word)
        if widx is not None:
            return widx
        else:
            return self.vocab_size + 1 # oo, +2 is for zero-padding


    def parse_json(self):

        for di, d in enumerate(self.data):

            start = time.time()

            print("Processing doc {}!".format(d['id']))
            d['candidates_orig'] = list(d['candidates'])

            # answers
            d['answer_id'] = d['candidates_orig'].index(d['answer'])

            # temp tokenization
            cand_temp = [self.tokenize(c) for c in d['candidates']]
            support_temp = [self.tokenize(s) for s in d['supports']]

            # connections between cand and doc
            d['doc2cand'] = []
            d['ment_pos'] = []
            d['cand_num'] = len(d['candidates'])
            d['doc_num'] = len(d['supports'])
            for si, s in enumerate(support_temp):
                for ci, c in enumerate(cand_temp):
                    for wi, w in enumerate(s):
                        if check(s, wi, c, self.stemmer):
                            if (si+d['cand_num'],ci) not in d['doc2cand']:
                                d['doc2cand'].append((si+self.max_cand, ci)) # candidates on top of documents
                                d['doc2cand'].append((ci, si+self.max_cand)) # to generate symmetric adj matrices

                            d['ment_pos'].append(ind(si,wi,ci,c)) # get mention positions

            # find mentions of query subject
            d['subject_pos'] = []
            query_temp = [wd for wd in self.tokenize(d['query'])]
            d['query_sub'] = query_temp[1:]
            if len(d['query_sub']) != 0:           
                occur_or_not, positions = Check_Entity_In_Doc(d['query_sub'], support_temp)
                d['subject_pos'].extend(positions)
                if len(d['subject_pos']) > self.max_sub_ment:
                    d['subject_pos'] = d['subject_pos'][:self.max_sub_ment]
            else:
                d['query_sub'] = self.tokenize(d['query'].replace('_', ' '))

            assert(len(d['subject_pos']) <= self.max_sub_ment)

            # connections among cands
            d['cand2cand'] = []
            for i in range(d['cand_num']):
                for j in range(d['cand_num']):
                    if i != j:
                        d['cand2cand'].append((i, j))

            # connections between doc and ment
            # note that the subject mentions are appened after candidate mentions, so in the model
            # the features of subject mentions will also be appened after candidate mentions to keep consistent
            d['doc2ment'] = []
            for idx, item in enumerate(d['ment_pos']):
                d['doc2ment'].append((item[0], idx + self.max_docs))
                d['doc2ment'].append((idx + self.max_docs, item[0])) # docs on top of mentions as nodes; this will be used in future.
                # plus max_docs because it's easier to slice out mention-only nodes
            for idx, item in enumerate(d['subject_pos']):
                d['doc2ment'].append((item[0], idx + self.max_docs+self.max_ment))
                d['doc2ment'].append((idx + self.max_docs+self.max_ment, item[0])) # docs on top of mentions as nodes; this will be used in future.
                # plus max_docs because it's easier to slice out mention-only nodes

            # connections among mentions
            edges_in, edges_out = [], []
            for idx0, m0 in enumerate(d['ment_pos'] + d['subject_pos']):
                for idx1, m1 in enumerate(d['ment_pos'] + d['subject_pos']):
                    if m0[0] == m1[0] and m0[1] != m1[1]: # same document and mention at different positions
                        if len(m0) == 3 and len(m1)==4: #m0 is subject pos and m1 not
                            edges_in.append((idx0+self.max_docs + self.max_ment-len(d['ment_pos']), idx1+self.max_docs))
                        if len(m0) == 4 and len(m1)==3:
                            edges_in.append((idx0+self.max_docs, idx1+self.max_docs+ self.max_ment-len(d['ment_pos'])))
                        if len(m0) == 4 and len(m1)==4:
                            edges_in.append((idx0+self.max_docs, idx1+self.max_docs))
                        if len(m0) == 3 and len(m1)==3:
                            edges_in.append((idx0+self.max_docs+ self.max_ment-len(d['ment_pos']), idx1+self.max_docs+ self.max_ment-len(d['ment_pos'])))

            for idx0, m0 in enumerate(d['ment_pos'] + d['subject_pos']):
                for idx1, m1 in enumerate(d['ment_pos'] + d['subject_pos']):
                    if m0[0] != m1[0] and m0[-1] == m1[-1]: # different document same candidates
                        if len(m0) == 3 and len(m1)==4: #m0 is subject pos and m1 not
                            edges_out.append((idx0+self.max_docs + self.max_ment-len(d['ment_pos']), idx1+self.max_docs))
                        if len(m0) == 4 and len(m1)==3:
                            edges_out.append((idx0+self.max_docs, idx1+self.max_docs+ self.max_ment-len(d['ment_pos'])))
                        if len(m0) == 4 and len(m1)==4:
                            edges_out.append((idx0+self.max_docs, idx1+self.max_docs))
                        if len(m0) == 3 and len(m1)==3:
                            edges_out.append((idx0+self.max_docs+ self.max_ment-len(d['ment_pos']), idx1+self.max_docs+ self.max_ment-len(d['ment_pos'])))

            d['edges_in'] = edges_in
            d['edges_out'] = edges_out

            #start = time()
            d['query'] = [self._w2id(wd) for wd in self.tokenize(d['query'].replace('_', ' '))]
            d['query_sub'] = [self._w2id(wd) for wd in d['query_sub']]   
            d['candidates'] = [[ self._w2id(wd) for wd in self.tokenize(c)] for c in d['candidates']]
            d['supports'] = [[ self._w2id(wd) for wd in self.tokenize(s)] for s in d['supports']]
            #print('Finish tokenization in {} seconds!'.format(time()-start))

            self.data[di] = d

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyCollator(object):

    def __init__(self, word_dropout=None, doc2cand = True, doc2ment = True, with_cand_edge = True, doc_edge=True, cand2ment_edge = True, 
        wd_ment_edge = True, ad_ment_edge = True, all_ment_edge = False, gnn_type = "gcn", max_sub = 10, fcgnn = False):
        if word_dropout == 0.0:
            self.word_dropout = None
        else:
            self.word_dropout = word_dropout
        self.doc2cand = doc2cand
        self.doc2ment = doc2ment
        self.wd_ment_edge = wd_ment_edge # whether to use within-doc edges for mentions
        self.ad_ment_edge = ad_ment_edge # whether to use across-doc edges for mentions
        self.with_cand_edge = with_cand_edge # whether to add edges among candidates
        self.doc_edge = doc_edge # whether to add edges among docs
        self.all_ment_edge = all_ment_edge # whether add edges for mentions other than WD and AD
        self.cand2ment_edge = cand2ment_edge
        self.gnn_type = gnn_type
        self.max_sub = max_sub
        self.fcgnn = fcgnn

    def _word_dropout(self, input, vocab_size):

        seq_len = len(input)
        samples = np.random.binomial(1,self.word_dropout,seq_len)
        output = []
        for i in range(seq_len):
            if samples[i] == 0:
                output.append(input[i])
        return output

    def __call__(self, data_mb):

        max_docs = 63
        max_cand = 79
        max_mention = 700
        max_sub_mention = self.max_sub
        max_nodes = 63 + 79 + 700 + max_sub_mention # docs + candidates + mentions + subs
        vocab_size = 427753

        start = time.time()

        def get_batch_stat(data_mb):
            max_query_len, max_doc_len, max_cand_len, max_cand_num, max_ment_num, max_doc_num = 0,0,0,0,0,0
            for d in data_mb:
                if len(d['query']) > max_query_len:
                    max_query_len = len(d['query'])
                if max([len(doc) for doc in d['supports']]) > max_doc_len:
                    max_doc_len = max([len(doc) for doc in d['supports']])
                if max([len(cand) for cand in d['candidates']]) > max_cand_len:
                    max_cand_len = max([len(cand) for cand in d['candidates']])

                if len(d['supports']) > max_doc_num:
                    max_doc_num = len(d['supports'])
                if len(d['ment_pos']) > max_ment_num:
                    max_ment_num = len(d['ment_pos'])
                if len(d['candidates']) > max_cand_num:
                    max_cand_num = len(d['candidates'])

            return max_query_len, max_doc_len, max_cand_len, max_cand_num, max_ment_num, max_doc_num

        max_query_len, max_doc_len, max_cand_len, max_cand_num, max_ment_num, max_doc_num = get_batch_stat(data_mb)

        # batching
        id_mb = [d['id'] for d in data_mb]
        adj_mb = []
        query_mb = np.ones((len(data_mb), max_query_len)) * (vocab_size-1)
        doc_mb = np.ones((len(data_mb), max_doc_num, max_doc_len))* (vocab_size-1) # for mention graph
        cand_mb = np.ones((len(data_mb), max_cand_num, max_cand_len))* (vocab_size-1)
        query_mb_len = np.zeros((len(data_mb),1))
        doc_mb_len = np.zeros((len(data_mb),max_doc_num))
        cand_mb_len = np.zeros((len(data_mb),max_cand_num))
        ment_pos_mb = np.zeros((len(data_mb), max_ment_num, 4))
        sub_pos_mb = np.zeros((len(data_mb), max_sub_mention, 3))
        ment2cand_mask = np.zeros((len(data_mb), max_cand, max_mention)) # correspondance between candidates and mention
        answer_candiates_id = np.zeros((len(data_mb),1))

        bmask_mb = np.zeros((len(data_mb), max_nodes)) # node mask for each graph

        candidate_mask = np.zeros((len(data_mb), max_cand))

        # nodes configuration: [cands, docs, mentions, subs]
        for di, d in enumerate(data_mb):

            if self.fcgnn:
                adj = np.ones((max_nodes, max_nodes)) - np.eye(max_nodes)
            else:
                adj_ = []

                # cand graph edges
                if self.doc_edge:
                    #print("Adding edges among docs")
                    doc2doc = np.zeros((max_nodes, max_nodes))
                    doc2doc[max_cand:max_cand+d['doc_num'], max_cand:max_cand+d['doc_num']] = 1 #matrix with all 1s
                    doc2doc = doc2doc - np.diagonal(doc2doc) # remove self connection
                    adj_.append(doc2doc)
                    
                if len(d['doc2cand']) == 0 or self.doc2cand == False:
                    adj_.append(np.zeros((max_nodes, max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['doc2cand'])), np.array(d['doc2cand']).T),
                        shape=(max_nodes, max_nodes)).toarray()

                    adj_.append(adj)

                if self.with_cand_edge:
                    if len(d['cand2cand']) == 0:
                        adj_.append(np.zeros((max_nodes, max_nodes)))
                    else:
                        adj = scipy.sparse.coo_matrix((np.ones(len(d['cand2cand'])), np.array(d['cand2cand']).T),
                            shape=(max_nodes, max_nodes)).toarray() 

                        adj_.append(adj)
                
                # doc to ment
                if len(d['doc2ment']) == 0 or self.doc2ment == False:
                    adj_.append(np.zeros((max_nodes, max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['doc2ment'])), np.array(d['doc2ment']).T + max_cand),
                        shape=(max_nodes, max_nodes)).toarray()

                    adj_.append(adj)

                # cand to ment:
                if self.cand2ment_edge:
                    if len(d['ment_pos']) == 0:
                        adj_.append(np.zeros((max_nodes, max_nodes)))
                    else:
                        cand2ment = []
                        for idx, ment in enumerate(d['ment_pos']):
                            cand2ment.append([idx+max_cand+max_docs, ment[-1]])
                            cand2ment.append([ment[-1], idx+max_cand+max_docs])
                        adj = scipy.sparse.coo_matrix((np.ones(len(cand2ment)), np.array(cand2ment).T),
                            shape=(max_nodes, max_nodes)).toarray()

                        adj_.append(adj)

                if len(d['edges_in']) == 0 or not self.wd_ment_edge:
                    adj_.append(np.zeros((max_nodes, max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_in'])), np.array(d['edges_in']).T + max_cand),
                        shape=(max_nodes, max_nodes)).toarray()

                    adj_.append(adj)

                if len(d['edges_out']) == 0 or not self.ad_ment_edge:
                    adj_.append(np.zeros((max_nodes, max_nodes)))
                else:
                    adj = scipy.sparse.coo_matrix((np.ones(len(d['edges_out'])), np.array(d['edges_out']).T + max_cand),
                        shape=(max_nodes, max_nodes)).toarray()   

                    adj_.append(adj)

                if self.all_ment_edge:
                    #print("Adding all_ment_edge!")
                    adj = np.zeros((max_nodes, max_nodes))
                    adj[max_cand + max_docs:, max_cand + max_docs:] = 1
                    temp = np.zeros((max_nodes, max_nodes))
                    temp[range(max_cand + max_docs, max_nodes), range(max_cand + max_docs, max_nodes)] = 1
                    adj = adj - adj_[-1] - adj_[-2] - temp
                    adj_.append(adj)

                adj = np.stack(adj_, 0)

            def adj_proc(adj):
                d_ = adj.sum(-1)
                d_[np.nonzero(d_)] **=  -1
                return adj * np.expand_dims(d_, -1)

            if self.gnn_type == "gcn" or self.gnn_type == "gat":
                adj_mb.append(adj_proc(adj))
            else:
                ajd_mb.append(adj)

            query_mb[di, :len(d['query'])] = d['query']
            query_mb_len[di] = len(d['query'])
            for doc_id, doc_seq in enumerate(d['supports']):
                if self.word_dropout is not None:
                    doc_seq_wddp = self._word_dropout(doc_seq, vocab_size) # apply word dropout
                    doc_mb[di, doc_id, :len(doc_seq_wddp)] = doc_seq_wddp
                    doc_mb_len[di, doc_id] = len(doc_seq_wddp)
                else:
                    doc_mb[di, doc_id, :len(doc_seq)] = doc_seq
                    doc_mb_len[di, doc_id] = len(doc_seq)

            for cand_id, cand_seq in enumerate(d['candidates']):
                cand_mb[di, cand_id, :len(cand_seq)] = cand_seq
                cand_mb_len[di, cand_id] = len(cand_seq)

            answer_candiates_id[di] = d['answer_id']

            bmask_mb[di, :d['cand_num']] = 1 # cand mask
            bmask_mb[di, max_cand:max_cand + d['doc_num']] = 1 # docs mask
            bmask_mb[di, max_cand + max_docs:max_cand + max_docs+len(d['ment_pos'])] = 1 # mentions mask
            bmask_mb[di, max_cand + max_docs + max_mention:max_cand + max_docs+max_mention+len(d['subject_pos'])] = 1 # mentions mask

            # an array indicates the start and end pos of each candidate in each document
            # The last dimension (size 4) includes (document index, start index, end index, candidate index)
            ment_pos_mb[di, :len(d['ment_pos']), :] = np.array(d['ment_pos'])
            if len(d['subject_pos']) > 0:
                sub_pos_mb[di, :len(d['subject_pos'])] = np.array(d['subject_pos'])

            for mid, x in enumerate(d['ment_pos']):
                ment2cand_mask[di, x[-1], mid] = 1

            # give candidates that don't have mention a padded node as their mentions
            for i in range(len(d['candidates_orig'])):
                if sum(ment2cand_mask[di,i,:]) == 0:
                    #print("Manually assign candidate {} a mention!".format(d['candidates_orig'][i]))
                    ment2cand_mask[di,i,len(d['ment_pos'])] = 1
                    bmask_mb[di,max_cand+max_docs+len(d['ment_pos'])] = 1

            # generate candidate mask
            candidate_mask[di,:len(d['candidates'])] = 1

        adj_mb = np.array(adj_mb)

        print('Batch prepared in {} seconds!'.format(time.time()-start))

        return {'id_mb': id_mb,
                'doc_mb': torch.tensor(doc_mb, dtype=torch.long),
                'doc_mb_len': torch.tensor(doc_mb_len, dtype=torch.int),
                'cand_mb':torch.tensor(cand_mb, dtype=torch.long),
                'cand_mb_len': torch.tensor(cand_mb_len, dtype=torch.int),
                'query_mb': torch.tensor(query_mb, dtype=torch.long),
                'query_mb_len': torch.tensor(query_mb_len, dtype=torch.int),
                'ment_pos_mb':torch.tensor(ment_pos_mb, dtype=torch.float),
                'sub_pos_mb':torch.tensor(sub_pos_mb, dtype=torch.float),
                'ment2cand_mask': torch.tensor(ment2cand_mask, dtype=torch.float),
                'bmask_mb': torch.tensor(bmask_mb, dtype=torch.float),
                'adj_mb': torch.tensor(adj_mb, dtype=torch.float),
                'answer_candiates_id': torch.tensor(answer_candiates_id, dtype=torch.long),
                'candidate_mask': torch.tensor(candidate_mask, dtype=torch.float)}


if __name__ == '__main__':
    
    dev_json_list = ["wikihop_comb/dev{}.json".format(i) for i in range(10)]

    dev_dataset = whDataset_comb(dev_json_list)
    dev_collator = MyCollator()
    dev_loader = DataLoader(dataset = dev_dataset, batch_size = 40, num_workers=0, collate_fn=dev_collator)

    for batch in dev_loader:
        print(batch['id_mb'])
        sys.exit()
