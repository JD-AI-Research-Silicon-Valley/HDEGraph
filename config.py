import argparse

def add_training_settings(parser):
    parser.add_argument('--batch-size', type=int, default=40,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--valid-check', type=int, default=1,
                        help='check validation set every N epochs')
    parser.add_argument('--seed', type=int, default=999,
                        help='random seed')
    parser.add_argument('--start-epoch', type=int, default=0,
                             help='initial training epoch')
    parser.add_argument('--coatt-scale', type=int, default=0, help='whether to use trainable scale for coattention')
    parser.add_argument('--cm-fusion', type=int, default=1, help='whether to fuse the score of cand and ment nodes')
    parser.add_argument('--adapt-fusion', type=int, default=1, help='whether to use trainable fusion parameter')
    parser.add_argument('--criterion', type=str, default="ce", help='training criterion')
    parser.add_argument('--lsmooth', type=float, default=0.1, help='factor for label smoothing')
    parser.add_argument('--no-cand', type=int, default=0, help='whether to include cand model output')
    parser.add_argument('--no-ment', type=int, default=0, help='whether to include ment model output')
    
    return parser

def add_gcn_settings(parser):
    parser.add_argument('--gnn-type', type=str, default="gcn", help='type of gnn')
    parser.add_argument('--num-sub', type=int, default="10", help='number of subject mentions to include')
    parser.add_argument('--num-hop', type=int, default=3,
                             help='number of gcn hop')
    parser.add_argument('--cand-edge', type=int, default=1,
                    help='whether to add links among candidates')
    parser.add_argument('--doc-edge', type=int, default=0,
                    help='whether to add links among docs')
    parser.add_argument('--doc2cand', type=int, default=1,
                    help='whether to add links among candidates')
    parser.add_argument('--doc2ment', type=int, default=1,
                    help='whether to add links among docs')
    parser.add_argument('--cdoc-edge', type=int, default=0,
                    help='whether to add links among docs for cand version')
    parser.add_argument('--mdoc-edge', type=int, default=0,
                    help='whether to add links among docs for mention version')
    parser.add_argument('--wd-ment-edge', type=int, default=1,
                    help='whether to add links among mentions within same doc')
    parser.add_argument('--ad-ment-edge', type=int, default=1,
                    help='whether to add links among mentions of the same entity in different doc')
    parser.add_argument('--all-ment-edge', type=int, default=0,
                    help='whether to add links among mentions other than ad and md')
    parser.add_argument('--cand2ment-edge', type=int, default=0,
                    help='whether to add links among cands and cands mentions')
    parser.add_argument('--gcn-dropout', type=float, default=0.0,
                        help='dropout rate for gcn. M=0 means no dropout applied')
    parser.add_argument('--alpha', type=float, default=1.0, help='fixed fusion weight')

    return parser

def add_ablation_exp(parser):
    parser.add_argument('--no-gnn', type=int, default=0, help='whether to use gnn')
    parser.add_argument('--fcgnn', type=int, default=0, help='whether to use fully connected gnn')

    return parser

def add_mul_worker_settings(parser):
    parser.add_argument('--num-gpu', type=int, default=0,
                        help='number of gpu cards used. 0 means no gpu is used')
    return parser

def add_learning_settings(parser):
    parser.add_argument('--finetune', type=int, default=0,
                        help='whether to continue training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--optim', default="adam", metavar='OPTIM',
                        help='optim used for training (sgd or adam)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate (default: 0.1). M=0 means no dropout applied')
    parser.add_argument('--embd-dp',type=float,default=0.0, help='dropout for embedding layer')
    parser.add_argument('--word-dropout', type=float, default=0.25,
                        help='word dropout rate (default: 0.25)')
    parser.add_argument('--batch-norm', type=int, default=0,
                        help='whether to use batch norm')

    parser.add_argument('--scheduler', type=str, default="plateau", help="which scheduler to use")

    parser.add_argument('--lr-patience',type=int,default=1) # for plateau scheduler
    parser.add_argument('--lr-reduction-factor',type=float,default=0.2) # for plateau scheduler or cosine scheduler
    parser.add_argument('--min-lr',type=float,default=1e-5) # for plateau scheduler or cosine scheduler

    parser.add_argument('--t-initial',type=int,default=3) # for cosine scheduler with restarts
    parser.add_argument('--t-mul',type=int,default=2) # for cosine scheduler with restarts

    return parser

def add_model_lstm_settings(parser):
    parser.add_argument('--rnn-size', type=int, default=50,
                        help='rnn size for the lstm model')
    parser.add_argument('--rnn-layer', type=int, default=1,
                        help='rnn layers for the lstm model')
    return parser

def add_io_settings(parser):
    parser.add_argument('--data-folder', default="",
                        help='data folder')
    parser.add_argument('--embd-matrix', default="",
                        help='embed matrix file')
    parser.add_argument('--model-dir', type=str, default="model/test",
			help='model saving directory')
    parser.add_argument('--input-file', default="", help = 'input json file for evaluation')
    parser.add_argument('--output-file', default="", help = 'output prediction for evaluation')
    return parser
