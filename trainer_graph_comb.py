import torch
import torch.nn as nn
from model_graph_comb import combGCN
import time, tqdm, sys, os, subprocess, math
from config import *
import numpy as np
from pathlib import Path
import pickle, time

from preprocessing_graph_comb import whDataset_comb, MyCollator
from torch.utils.data import DataLoader

from allennlp_lr_scheduler import CosineWithRestarts
from label_smooth_criterion import LabelSmoothingLoss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def batch2cuda(batch, device):

    batch_size = batch['cdoc_mb'].shape[0]

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

def run_epoch(model, parallel_model, optimizer, criterion, train_ldr, device, device_ids, it, avg_loss, e):

    for batch in train_ldr:

        start_t = time.time()
        optimizer.zero_grad()

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

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm(model.parameters(), 100)
        loss = loss.data

        optimizer.step()
        end_t = time.time()
        model_t = end_t - start_t

        exp_w = 0.0  # smoothing
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        print("Train Epoch {}, iteration {}, loss {}, in {} seconds, lr {}".format(e, it, avg_loss, model_t, get_lr(optimizer)))
        it += 1

        torch.cuda.empty_cache()

    return it, avg_loss


def eval_dev(model, parallel_model, criterion, ldr, device, device_ids, taskid, model_dir=None):
    losses = [];
    all_preds = [];
    all_labels = []

    with torch.no_grad():

        model.eval()

        if model_dir:
            checkpoint = torch.load(Path(model_dir))
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

    if taskid == "Dev":
        model.train()

    _, max_index = torch.max(torch.stack(all_preds), 1)
    acc = float((max_index == torch.stack(all_labels).long()).sum()) / len(all_labels)

    loss = sum(losses) / len(losses)
    print("{} Loss {:.3f}, Acc {:.3f}".format(taskid, loss, acc))

    return loss, acc


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
                    max_sub = args.num_sub, no_gnn = (args.no_gnn==1), alpha = args.alpha, embd_matrix = args.embd_matrix)
    
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

    train_json_list = ["{}/train{}.json".format(args.data_folder, i) for i in range(10)]
    dev_json_list = ["{}/dev{}.json".format(args.data_folder, i) for i in range(10)]

    train_data = whDataset_comb(train_json_list)
    dev_data = whDataset_comb(dev_json_list)
    train_collator = MyCollator(args.word_dropout, args.doc2cand==1, args.doc2ment==1, args.cand_edge==1, args.doc_edge == 1, args.cand2ment_edge==1, args.wd_ment_edge==1, 
                                args.ad_ment_edge==1, args.all_ment_edge==1, gnn_type = args.gnn_type, max_sub = args.num_sub, fcgnn=(args.fcgnn==1))
    dev_collator = MyCollator(doc2cand = (args.doc2cand==1), doc2ment = (args.doc2ment==1), wd_ment_edge=(args.wd_ment_edge==1), ad_ment_edge=(args.ad_ment_edge==1), \
                                with_cand_edge=(args.cand_edge==1), doc_edge = (args.doc_edge==1),
                                all_ment_edge = (args.all_ment_edge==1), cand2ment_edge = (args.cand2ment_edge==1), gnn_type = args.gnn_type, 
                                max_sub = args.num_sub, fcgnn = (args.fcgnn==1))
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, num_workers=0, shuffle=True, collate_fn=train_collator)
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

    run_state = (0, 0)
    best_so_far = float("-inf")
    for e in range(args.start_epoch, args.epochs):
        start = time.time()

        run_state = run_epoch(model, parallel_model, optimizer, criterion, train_loader, device, device_ids, *run_state, e)

        msg = "Train Epoch {} completed in {:.2f} (s), AVG loss: {}."
        print(msg.format(e, time.time() - start, run_state[1]))

        run_state = (0, 0)

        if e % args.valid_check == 0:
            dev_loss, dev_acc = eval_dev(model, parallel_model, criterion, dev_loader, device, device_ids, "Dev")
            if args.scheduler == 'plateau':
                scheduler.step(round(float(dev_loss.cpu().numpy()), 2))
            else:
                scheduler.step()
            # Save the best model on the dev set
            if dev_acc > best_so_far:
                print("---------Saving model---------")
                best_so_far = dev_acc
                torch.save(model.state_dict(), args.model_dir)

    print('Best accuracy on dev set is {}!'.format(best_so_far))


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
