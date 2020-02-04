import torch
import torch.nn as nn
import torch.nn.functional as F

from sort_helper import SeqSortHelper
import pickle, sys

from coattn import CoAttn

MIN_VAL = -99999999


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n, d = x.size(0), x.size(1), x.size(2)
        x = x.contiguous().view(t * n, d)
        x = self.module(x)
        x = x.contiguous().view(t, n, d)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, dropout=0.0, bidirectional=False, rnn_type=nn.LSTM,
                 batch_norm=False):
        super(MaskRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, num_layers=rnn_layers,
                            bidirectional=bidirectional, bias=True, dropout=dropout)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, total_len):
        # Input: seqLength X batchSize X dim
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, total_length=total_len)
        # h, _ = nn.utils.rnn.pad_packed_sequence(h)
        # if self.bidirectional:
        #     x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x, h


class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, rnn_layers, rnn_type=nn.GRU, dropout=0.1, bidirectional=True, batch_norm = True):
        super(RNNNet, self).__init__()
        self.sorthelper = SeqSortHelper()
        self.model = MaskRNN(input_dim, hidden_dim, rnn_layers, rnn_type=rnn_type, dropout=dropout, bidirectional=bidirectional, batch_norm=batch_norm)
        self.bidirectional = bidirectional

    # input: 3d tensor (seq_len, batch_size, fea_dim)
    #       It could be a PackedSequence to included seq_len for each seqence
    def forward(self, input, input_lengths):

        seq_len, bs, dim = input.size()

        sorted_input, sorted_input_len, perm_idx_back = self.sorthelper.sort_input(input, input_lengths)
        num_non_zero = torch.nonzero(sorted_input_len).size(0)

        output, state = self.model(sorted_input.narrow(1,0,num_non_zero), sorted_input_len.narrow(0,0,num_non_zero), seq_len)

        output_dim = output.size(-1)
        state_dim = state.size(-1)

        output = torch.cat((output, torch.zeros(seq_len, bs-num_non_zero, output_dim).to(input.device)), dim=1)
        state = torch.cat((state, torch.zeros(2, bs-num_non_zero, state_dim).to(input.device)), dim=1)

        output = self.sorthelper.restore_order_input(output, perm_idx_back)
        state = self.sorthelper.restore_order_input(state, perm_idx_back)

        return output, state

class gcnLayer(nn.Module):
    def __init__(self, input_dim, proj_dim=512, dropout=0.1, num_hop=3, gcn_num_rel=2, batch_norm=False):
        super(gcnLayer, self).__init__()
        self.proj_dim = proj_dim
        self.num_hop = num_hop
        self.gcn_num_rel = gcn_num_rel

        for i in range(gcn_num_rel):
            setattr(self, "fr{}".format(i+1), nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False)))

        self.fs = nn.Sequential(nn.Linear(input_dim, proj_dim), nn.Dropout(dropout, inplace=False))

        self.fa = nn.Sequential(nn.Linear(input_dim + proj_dim, proj_dim), nn.Dropout(dropout, inplace=False))

    def forward(self, input, input_mask, adj):
        # input: bs x max_nodes x node_dim
        # input_mask: bs x max_nodes
        # adj: bs x 3 x max_nodes x max_nodes
        # num_layer: number of layers; note that the parameters of all layers are shared

        cur_input = input.clone()

        for i in range(self.num_hop):
            # integrate neighbor information
            nb_output = torch.stack([getattr(self, "fr{}".format(i+1))(cur_input) for i in range(self.gcn_num_rel)],
                                    1) * input_mask.unsqueeze(-1).unsqueeze(1)  # bs x 2 x max_nodes x node_dim

            # apply different types of connections, which are encoded in adj matrix
            update = torch.sum(torch.matmul(adj,nb_output), dim=1, keepdim=False) + \
                     self.fs(cur_input) * input_mask.unsqueeze(-1)  # bs x max_node x node_dim

            # get gate values
            gate = torch.sigmoid(self.fa(torch.cat((update, cur_input), -1))) * input_mask.unsqueeze(
                -1)  # bs x max_node x node_dim

            # apply gate values
            cur_input = gate * torch.tanh(update) + (1 - gate) * cur_input  # bs x max_node x node_dim

        return cur_input

class combGCN(nn.Module):
    def __init__(self, max_cand = 79, max_doc = 63, max_ment = 700, max_sub=10, feat_dim=400, embd_dp = 0.1,
                 dropout=0.1, rnn_layer=1, rnn_size=50, gcn_hop=3, batch_norm = False, adapt_scale = True, gcn_num_rel=4, 
                 gcn_dropout=0.0, cm_fusion = True, adapt_fusion=True, gnn_type = "gcn", no_gnn=False, alpha = 1.0, embd_matrix = None):
        super(combGCN, self).__init__()

        self.no_gnn = no_gnn

        # graph parameters
        self.max_cand = max_cand
        self.max_doc = max_doc
        self.max_ment = max_ment
        self.max_sub = max_sub
        self.feat_dim = feat_dim

        self.embd_weight = torch.load(embd_matrix)
        self.vocab_size = self.embd_weight.size(0)

        # embedding layer
        self.embedding = torch.nn.Embedding(self.vocab_size, self.feat_dim, padding_idx=self.vocab_size-1)
        self.embedding.weight.data.copy_(self.embd_weight)
        self.embedding.weight.requires_grad = False # freeze the embedding weight
        self.embd_dropout = nn.Dropout(embd_dp, inplace=False)

        # encoder
        self.query_encoder = RNNNet(self.feat_dim, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)
        self.doc_encoder = RNNNet(self.feat_dim, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)
        self.cand_encoder = RNNNet(self.feat_dim, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)

        # co-attention
        self.query_doc_coatt = CoAttn(self.feat_dim, adapt_scale=adapt_scale)
        self.doc_coatt_rnn = RNNNet(2*rnn_size, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)
        self.query_ment_coatt = CoAttn(self.feat_dim, adapt_scale=adapt_scale)
        self.ment_coatt_rnn = RNNNet(2*rnn_size, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)
        self.query_cand_coatt = CoAttn(self.feat_dim, adapt_scale=adapt_scale)
        self.cand_coatt_rnn = RNNNet(2*rnn_size, rnn_size, rnn_layer, dropout=dropout, rnn_type=nn.GRU, batch_norm=batch_norm)

        # self attention
        self.doc_selfatt = nn.Sequential(nn.Linear(4*rnn_size, 2*rnn_size), nn.Tanh(), 
            nn.Dropout(dropout, inplace=False), nn.Linear(2*rnn_size, 1), nn.Tanh())
        self.ment_selfatt = nn.Sequential(nn.Linear(4*rnn_size, 2*rnn_size), nn.Tanh(), 
            nn.Dropout(dropout, inplace=False), nn.Linear(2*rnn_size, 1), nn.Tanh())
        self.cand_selfatt = nn.Sequential(nn.Linear(4*rnn_size, 2*rnn_size), nn.Tanh(), 
            nn.Dropout(dropout, inplace=False), nn.Linear(2*rnn_size, 1), nn.Tanh())
        self.sub_selfatt = nn.Sequential(nn.Linear(2*rnn_size, rnn_size), nn.Tanh(), 
            nn.Dropout(dropout, inplace=False), nn.Linear(rnn_size, 1), nn.Tanh())
        self.sub_proj = nn.Sequential(nn.Linear(2*rnn_size, 4*rnn_size), nn.Tanh()) # proj sub features to higher dimension
        self.sfm = nn.Softmax(-1)

        if gnn_type == "gcn":
            self.gcn = gcnLayer(input_dim = 4*rnn_size, proj_dim = 4*rnn_size, num_hop=gcn_hop, gcn_num_rel=gcn_num_rel, dropout=gcn_dropout)
        if gnn_type == "gin":
            print("Using GIN layer!")
            self.gcn = ginLayer(input_dim = 4*rnn_size, proj_dim = 4*rnn_size, num_hop=gcn_hop, num_rel=gcn_num_rel, dropout=gcn_dropout)
        if gnn_type == "gat":
            print("Using GAT layer!")
            self.gcn = gatLayer(input_dim = 4*rnn_size, proj_dim = 4*rnn_size, num_hop=gcn_hop, num_rel=gcn_num_rel, dropout=gcn_dropout)
        
        self.cand_output_FC = nn.Sequential(nn.Linear(4*rnn_size, 2*rnn_size), nn.Tanh(), nn.Dropout(dropout,inplace=False),
                                       nn.Linear(2*rnn_size, 1))
        self.ment_output_FC = nn.Sequential(nn.Linear(4*rnn_size, 2*rnn_size), nn.Tanh(), nn.Dropout(dropout,inplace=False),
                                       nn.Linear(2*rnn_size, 1))
        self.cm_fusion = cm_fusion
        if adapt_fusion:
            self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float))
        else:
            self.alpha = alpha

    def gen_mask(self, max_len, lengths, device):
        lengths = lengths.type(torch.LongTensor)
        num = lengths.size(0)
        vals = torch.LongTensor(range(max_len)).unsqueeze(0).expand(num, -1)+1 # +1 for masking out sequences with length 0
        mask = torch.gt(vals, lengths.unsqueeze(1).expand(-1, max_len)).to(device)
        return mask

    def do_coattn(self, query_feat, query_len, node_feat, node_len, coattn, coattn_rnn):
        #query_feat: max_len x bs x feat_dim
        #node_feat: max_len x (bs x max_doc) x feat_dim

        #len_temp = query_len.unsqueeze(-1).expand(32,35).contiguous().view(32*35) # for debuging the behavior of torch.expand

        query_mask = self.gen_mask(query_feat.size(0), query_len, query_feat.device)
        node_mask = self.gen_mask(node_feat.size(0), node_len, node_feat.device)

        output_ss, output_sq, cs_input = coattn(query_feat, node_feat, query_mask, node_mask)

        cs_output = coattn_rnn.forward(cs_input, node_len)[0]
        
        output = torch.cat((cs_output, output_ss), dim=-1)

        return output

    # self attentive pooling
    def do_selfatt(self, input, input_len, selfatt):

        # input: max_len X batch_size X dim

        input_mask = self.gen_mask(input.size(0), input_len, input.device)

        att = selfatt.forward(input).squeeze(-1).transpose(0,1)
        att = att.masked_fill(input_mask, MIN_VAL)
        att_sfm = self.sfm(att).unsqueeze(1)

        output = torch.bmm(att_sfm, input.transpose(0,1)).squeeze(1) # batchsize x dim

        return output

    def extract_ment_embd(self, doc_embd, doc_len, batch_size, num_doc, max_ment, max_ment_len, ment_pos):

        max_len, _, d_embd = doc_embd.size()
        doc_embd_rs = doc_embd.transpose(0,1).contiguous().view(batch_size, num_doc, max_len, d_embd)
        doc_len = doc_len.view(batch_size, num_doc)
        max_ment_num = ment_pos.size(1)

        ment_embd_len = torch.zeros(batch_size, max_ment).to(doc_embd.device)
        ment_embd_len[:, :max_ment_num] = ment_pos[:,:,2]-ment_pos[:,:,1]
        ment_embd = torch.zeros(batch_size, max_ment, max_ment_len, d_embd).to(doc_embd.device)
        for bi in range(batch_size):
            for mi in range(max_ment_num):
                doc_idx = int(ment_pos[bi,mi,0])
                start_idx = int(ment_pos[bi,mi,1])
                end_idx = int(ment_pos[bi,mi,2])

                ment_embd[bi,mi,:(end_idx-start_idx),:] = doc_embd_rs[bi,doc_idx,start_idx:end_idx,:]

        ment_embd = ment_embd.view(batch_size*max_ment,max_ment_len,d_embd).contiguous().transpose(0,1)
        ment_embd_len = ment_embd_len.contiguous().view(batch_size*max_ment)

        return ment_embd, ment_embd_len

    def forward(self, doc_feat, doc_len, query_feat, query_len, cand_feat, cand_len, \
                     ment_pos, sub_ment_pos, ment2cand_mask, adj, bmask):

        doc_feat = self.embd_dropout(self.embedding(doc_feat))
        query_feat = self.embd_dropout(self.embedding(query_feat))
        cand_feat = self.embd_dropout(self.embedding(cand_feat))

        # encode candidates embeddings
        bs, b_cnum, b_clen, feat_dim = cand_feat.size() # batch_size x max number of docs of all samples x max len of all docs in this batch x feat_dim
        cand_feat_rnn = cand_feat.view(bs*b_cnum, b_clen, feat_dim).contiguous().transpose(0,1)
        cand_len_rnn = cand_len.view(bs*b_cnum)
        cand_encoder_output = self.cand_encoder.forward(cand_feat_rnn, cand_len_rnn)[0]

        # encode document embeddings
        bs, b_dnum, b_dlen, feat_dim = doc_feat.size() # batch_size x max number of docs of all samples x max len of all docs in this batch x feat_dim
        doc_feat_rnn = doc_feat.view(bs*b_dnum, b_dlen, feat_dim).contiguous().transpose(0,1)
        doc_len_rnn = doc_len.view(bs*b_dnum)
        doc_encoder_output = self.doc_encoder.forward(doc_feat_rnn, doc_len_rnn)[0]
        
        # extract mentions embeddings
        max_ment_len = int(torch.max(ment_pos[:,:,2]-ment_pos[:,:,1]))
        ment_encoder_output, ment_len_rnn = self.extract_ment_embd(doc_encoder_output, doc_len_rnn, bs, b_dnum, self.max_ment, max_ment_len, ment_pos)
        ment_encoder_output = ment_encoder_output.to(doc_feat.device)
        ment_len_rnn = ment_len_rnn.to(doc_len.device)

        # extract subject mentions embeddings
        max_sub_len = int(torch.max(sub_ment_pos[:,:,2]-sub_ment_pos[:,:,1]))
        sub_encoder_output, sub_len_rnn = self.extract_ment_embd(doc_encoder_output, doc_len_rnn, bs, b_dnum, self.max_sub, max_sub_len, sub_ment_pos)
        sub_encoder_output = sub_encoder_output.to(doc_feat.device)
        sub_len_rnn = sub_len_rnn.to(doc_len.device)    

        # encode query embeddings
        query_feat_rnn = query_feat.transpose(0,1)
        query_len_rnn = query_len.view(bs)
        query_encoder_output = self.query_encoder.forward(query_feat_rnn, query_len_rnn)[0]

        # coattention between query and documents
        qd_coatt_output= self.do_coattn(query_encoder_output, query_len_rnn,
             doc_encoder_output, doc_len_rnn, self.query_doc_coatt, self.doc_coatt_rnn)
        #coattention between query and mentions
        qm_coatt_output = self.do_coattn(query_encoder_output, query_len_rnn,
             ment_encoder_output, ment_len_rnn, self.query_ment_coatt, self.ment_coatt_rnn)
        #coattention between query and candidates
        qc_coatt_output = self.do_coattn(query_encoder_output, query_len_rnn,
             cand_encoder_output, cand_len_rnn, self.query_cand_coatt, self.cand_coatt_rnn)

        # self attention of query-documents embeddings
        qd_self_output = self.do_selfatt(qd_coatt_output, doc_len_rnn, self.doc_selfatt) # (bs X max_doc_of_current_batch) X 200
        # self attention of query-mention embeddings
        qm_self_output = self.do_selfatt(qm_coatt_output, ment_len_rnn, self.ment_selfatt) # (bs X max_cand_of_current_batch) X 200
        # self attention of query-candidates embeddings
        qc_self_output = self.do_selfatt(qc_coatt_output, cand_len_rnn, self.cand_selfatt) # (bs X max_cand_of_current_batch) X 200
        # self attention of subject mention embeddings
        sub_self_output = self.do_selfatt(sub_encoder_output, sub_len_rnn, self.sub_selfatt)
        sub_self_output = self.sub_proj(sub_self_output) # increase dimension 

        # build graph node representations
        feat_dim = qm_self_output.size(-1)
        qm_node_feat = qm_self_output.view(bs, self.max_ment, feat_dim)
        sub_node_feat = sub_self_output.view(bs, self.max_sub, feat_dim)

        qc_node_feat = qc_self_output.view(bs, b_cnum, feat_dim)
        qc_node_feat = torch.cat((qc_node_feat, torch.zeros(bs, self.max_cand-b_cnum, feat_dim).to(qc_node_feat.device)), dim=1) # pad to self.max_cand
        qd_node_feat = qd_self_output.view(bs, b_dnum, feat_dim)
        qd_node_feat = torch.cat((qd_node_feat, torch.zeros(bs, self.max_doc-b_dnum, feat_dim).to(qd_node_feat.device)), dim=1) # pad to self.max_doc

        # concatenate mentions and documents representations. Documents on top of mentions!
        node_feat = torch.cat((qc_node_feat, qd_node_feat, qm_node_feat, sub_node_feat), dim=1) # bs X max_node X feat_dim
        assert(node_feat.size(1)==adj.size(-1)) # make sure number of nodes matches

        if self.no_gnn:
            gcn_output = node_feat
        else:
            gcn_output = self.gcn(node_feat, bmask, adj)* bmask.unsqueeze(-1)

        gcn_cand_output = gcn_output.narrow(1,0,self.max_cand) # take cand nodes
        cand_final_output = self.cand_output_FC(gcn_cand_output)
        cand_final_output = torch.squeeze(cand_final_output, -1)
        # masking
        cand_bmask = bmask.narrow(-1,0, self.max_cand)
        cand_final_output = cand_final_output * cand_bmask
        # replace 0 with -inf         
        cand_final_output = torch.where(torch.eq(cand_final_output, torch.zeros(cand_final_output.size()).to(node_feat.device)), 
                                   (torch.ones(cand_final_output.size()) * float(-1e30)).to(node_feat.device),         
                                   cand_final_output)
        if self.cm_fusion:
            gcn_ment_output = gcn_output.narrow(1,self.max_cand+self.max_doc,self.max_ment) # take mentions nodes

            ment_final_output = self.ment_output_FC(gcn_ment_output)

            ment_final_output = torch.squeeze(ment_final_output, -1)  # bs x max_ment

            # masking
            ment_bmask = bmask.narrow(-1,self.max_cand+self.max_doc, self.max_ment)
            ment_final_output = ment_final_output * ment_bmask

            # fusing
            ment_final_output = ment2cand_mask * torch.unsqueeze(ment_final_output, 1) # bs X max_cand X max_mention
            
            # replace 0 with -inf         
            ment_final_output = torch.where(torch.eq(ment_final_output, torch.zeros(ment_final_output.size()).to(node_feat.device)), 
                                       (torch.ones(ment_final_output.size()) * float(-1e30)).to(node_feat.device),         
                                       ment_final_output)
            # reduce_max
            ment_final_output = torch.max(ment_final_output, -1, keepdim=False)[0]

        if self.cm_fusion:
            return self.alpha * cand_final_output + ment_final_output
        else:
            return cand_final_output

    def get_param_size(self):
        params = 0
        for m in [self.query_encoder, self.doc_encoder, self.cand_encoder, self.doc_coatt_rnn, self.ment_coatt_rnn, \
                    self.cand_coatt_rnn, self.cand_selfatt,
                    self.doc_selfatt, self.ment_selfatt, self.gcn, self.cand_output_FC, self.ment_output_FC]:
            if m:
                for p in m.parameters():
                    tmp = 1
                    for x in p.size():
                        tmp *= x
                    params += tmp
        return params

if __name__ == '__main__':

    torch.manual_seed(123)

    dev_json_list = ["wikihop_comb/dev{}.json".format(i) for i in range(10)]

    model = docGCN()

    dev_dataset = whDataset_cand(dev_json_list)
    dev_loader = whDataLoader_cand(dataset = dev_dataset, batch_size = 80, num_workers=0, shuffle=True)

    for batch in dev_loader:
        
        predict = model(batch['doc_mb'], batch['doc_mb_len'], batch['query_mb'], batch['query_mb_len'], batch['cand_mb'], batch['cand_mb_len'], batch['adj_mb'], batch['bmask_mb'])

        sys.exit()












