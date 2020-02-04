import torch
import torch.nn as nn
from model_graph_comb import combGCN
import time, tqdm, sys, os, subprocess, math, json
from config import *
import numpy as np
from pathlib import Path
import pickle, time

from preprocessing_graph_comb import whDataset_comb, MyCollator
from torch.utils.data import DataLoader

from allennlp_lr_scheduler import CosineWithRestarts
from label_smooth_criterion import LabelSmoothingLoss

from collections import OrderedDict


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch2cuda(batch, device):

    batch_size = batch['doc_mb'].shape[0]

    batch['doc_mb'] = batch['doc_mb'].to(device)
    batch['doc_mb_len'] = batch['doc_mb_len'].to(device)
    batch['cand_mb'] = batch['cand_mb'].to(device)
    batch['cand_mb_len'] = batch['cand_mb_len'].to(device)
    batch['query_mb'] = batch['query_mb'].to(device)
    batch['query_mb_len'] = batch['query_mb_len'].to(device)
    batch['adj_mb'] = batch['adj_mb'].to(device)
    batch['bmask_mb'] = batch['bmask_mb'].to(device)
    batch['ment_pos_mb'] = batch['ment_pos_mb'].to(device)
    batch['sub_pos_mb'] = batch['sub_pos_mb'].to(device)
    batch['ment2cand_mask'] = batch['ment2cand_mask'].to(device)
    batch['answer_candiates_id'] = batch['answer_candiates_id'].view(batch_size).to(device)
    batch['candidate_mask'] = batch['candidate_mask'].to(device)

    return batch

def batch2mcuda(batch):

    batch_size = batch['doc_mb'].shape[0]

    # batch['doc_mb'] = torch.FloatTensor(batch['doc_mb'])
    # batch['doc_mb_len'] = torch.IntTensor(batch['doc_mb_len'])
    # batch['cand_mb'] = torch.FloatTensor(batch['cand_mb'])
    # batch['cand_mb_len'] = torch.IntTensor(batch['cand_mb_len'])
    # batch['query_mb'] = torch.FloatTensor(batch['query_mb'])
    # batch['query_mb_len'] = torch.IntTensor(batch['query_mb_len'])
    # batch['adj_mb'] = torch.FloatTensor(batch['adj_mb'])
    # batch['bmask_mb'] = torch.FloatTensor(batch['bmask_mb'])
    batch['answer_candiates_id'] = batch['answer_candiates_id'].view(batch_size)

    return batch

def eval_dev(model, parallel_model, criterion, ldr, device, device_ids, taskid, model_dir=None):
    losses = [];
    all_preds = []
    all_labels = []

    with torch.no_grad():

        model.eval()

        if model_dir:
            checkpoint = torch.load(Path(model_dir), map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
            print("---------{} model loaded!---------".format(taskid))

        for batch in ldr:

            if not device_ids:
                batch = batch2cuda(batch, device)
            else:
                batch = batch2mcuda(batch) # multiple GPU
                batch['answer_candiates_id'] = batch['answer_candiates_id'].to(device)
                batch['candidate_mask'] = batch['candidate_mask'].to(device)

            if parallel_model is not None:
                preds = parallel_model.forward(batch['doc_mb'], batch['doc_mb_len'], batch['query_mb'], batch['query_mb_len'], \
                                            batch['cand_mb'], batch['cand_mb_len'], batch['ment_pos_mb'], batch['sub_pos_mb'], batch['ment2cand_mask'], \
                                            batch['adj_mb'], batch['bmask_mb'])
            else:
                preds = model.forward(batch['doc_mb'], batch['doc_mb_len'], batch['query_mb'], batch['query_mb_len'], \
                                            batch['cand_mb'], batch['cand_mb_len'], batch['ment_pos_mb'], batch['sub_pos_mb'], batch['ment2cand_mask'], \
                                            batch['adj_mb'], batch['bmask_mb'])
            if type(criterion) is nn.CrossEntropyLoss:
                loss = criterion(preds, batch['answer_candiates_id'].long())
            else:
                loss = criterion(preds, batch['answer_candiates_id'].long(), batch['candidate_mask'])
            losses.append(loss.data)
            all_preds.extend(preds.detach())
            all_labels.extend(batch['answer_candiates_id'].detach())

            torch.cuda.empty_cache()

    _, max_index = torch.max(torch.stack(all_preds), 1)
    acc = float((max_index == torch.stack(all_labels).long()).sum()) / len(all_labels)

    return max_index.cpu().numpy().tolist(), acc

def get_json_output(dev_data, pred_idx, output_file):

    dev_pred = OrderedDict()
    for di, d in enumerate(dev_data.data):
        dev_pred[d['id']] = d['candidates_orig'][pred_idx[di]]

    with open(output_file, 'w') as fid:
        json.dump(dev_pred, fid)


def train(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_ids = None
    if args.num_gpu >= 1:
        print("Using GPU!")
        if args.num_gpu > 1:
            device_ids = range(args.num_gpu)
        aa = []
        for i in range(args.num_gpu - 1, -1, -1):
            device = torch.device("cuda:%d" % i)
            aa.append(torch.rand(1).to(device))  # a place holder
    else:
        device = 'cpu'
        print("Using CPU!")

    if args.fcgnn:
        gcn_num_rel = 1 # if using fully connected gnn, there is only one connection
    else:
        gcn_num_rel = 2 + args.cand_edge + args.doc_edge + args.wd_ment_edge + args.ad_ment_edge + args.all_ment_edge + args.cand2ment_edge

    model = combGCN(embd_dp = args.embd_dp, dropout=args.dropout, rnn_size=args.rnn_size, rnn_layer=args.rnn_layer, gcn_hop=args.num_hop, 
                    batch_norm = (args.batch_norm==1), adapt_scale = (args.coatt_scale==1), 
                    gcn_num_rel = gcn_num_rel,
                    gcn_dropout = args.gcn_dropout, cm_fusion = (args.cm_fusion==1), adapt_fusion=(args.adapt_fusion==1), gnn_type = args.gnn_type,
                    max_sub = args.num_sub, alpha = args.alpha, embd_matrix = args.embd_matrix)
    
    if args.criterion == 'ce':
        print("Using cross entropy loss!")
        criterion = nn.CrossEntropyLoss()
    else:
        print("Using label smoothing loss!")
        criterion = LabelSmoothingLoss(model.max_cand, label_smoothing=args.lsmooth)

    if os.path.exists(args.model_dir) and args.finetune == 1:
        print("Continue training on {}".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))

    model = model.to(device)
    model.use_cuda = True
    criterion = criterion.to(device)
    parallel_model = None
    if args.num_gpu > 1:
        print("Using multiple GPUs {}".format(device_ids))
        parallel_model = nn.parallel.DataParallel(model, device_ids, device)

    parameters = model.parameters()
    print(model)
    print("Number of parameters: {}".format(model.get_param_size()))
    if args.optim == "adam":
        print("Using Adam optimizer!")
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    else:
        print("Using SGD optimizer!")
        optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                    momentum=args.momentum, nesterov=True)

    dev_data = whDataset_comb(args.input_file, 'src/vocab.pkl', args.num_sub)
    dev_collator = MyCollator(wd_ment_edge=(args.wd_ment_edge==1), ad_ment_edge=(args.ad_ment_edge==1), \
                                with_cand_edge=(args.cand_edge==1), doc_edge = (args.doc_edge==1),
                                all_ment_edge = (args.all_ment_edge==1), cand2ment_edge = (args.cand2ment_edge==1), gnn_type = args.gnn_type, 
                                max_sub = args.num_sub)
    dev_loader = DataLoader(dataset = dev_data, batch_size = args.batch_size, num_workers=0, shuffle=False, collate_fn=dev_collator)


    if args.scheduler == 'plateau':
        print("Using learning rate scheduler based on dev loss!")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   'min',
                                                   factor=args.lr_reduction_factor,
                                                   patience=args.lr_patience,
                                                   verbose=False,
                                                   min_lr=args.min_lr)
    elif args.scheduler == 'cosine':
        print("Using cosine annealing with restarts scheduler!")
        scheduler = CosineWithRestarts(optimizer, T_max=args.t_initial, factor=args.t_mul, eta_min=args.min_lr)

    dev_pred_idx, dev_acc = eval_dev(model, parallel_model, criterion, dev_loader, device, device_ids, "Dev", model_dir = args.model_dir)

    print('Accuracy on dev is {}'.format(dev_acc))

    get_json_output(dev_data, dev_pred_idx, args.output_file)


def main():
    parser = argparse.ArgumentParser(description='PyTorch docGCN trainer')

    # Model configure settings
    add_mul_worker_settings(parser)

    # GCN configure settings
    add_gcn_settings(parser)

    # lstm setup if m-type=lstm is selected
    add_model_lstm_settings(parser)

    add_training_settings(parser)
    add_learning_settings(parser)

    add_io_settings(parser)

    add_ablation_exp(parser)

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
