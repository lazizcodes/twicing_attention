import argparse
import time
import math
import os, sys
import itertools
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel

# quick args
parser = argparse.ArgumentParser(
    description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--carry_over_fast_weight', action='store_true',
                    help='carry over fast weights.')
parser.add_argument('--eval_batch_size', type=int, default=10,
                    help='evaluation batch size')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')

args = parser.parse_args()

device = torch.device('cuda' if args.cuda else 'cpu')

### build dataset ###
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

### build model ###
cutoffs, tie_projs = [], [False]
cutoffs = [20000, 40000, 200000]
tie_projs += [True] * len(cutoffs)

model = MemTransformerLM(
        ntokens,
        args.n_layer,
        args.n_head,
        args.d_model,
        args.d_head,
        args.d_inner,
        args.dropout,
        args.dropatt,
        M_positions = args.M_positions,
        show_M = args.show_M,
        tie_weight=args.tied,
        d_embed=args.d_embed,
        div_val=args.div_val,
        tie_projs=tie_projs,
        pre_lnorm=args.pre_lnorm,
        tgt_len=args.tgt_len,
        ext_len=args.ext_len,
        mem_len=args.mem_len,
        cutoffs=cutoffs,
        same_length=args.same_length,
        attn_type=args.attn_type,
        clamp_len=args.clamp_len,
        sample_softmax=args.sample_softmax,
        proj_dim=args.performer_proj_dim,
        n_roll=args.dpfp_n_roll,
        skip_attn_normalization=args.skip_attn_normalization,
        no_pos=args.no_pos,
        device=device,
        update_mode=args.update_mode,
        kernel_size=args.kernel_size, 
        stride=args.stride,
        n_global_head=args.n_global_head
    )

with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
    print('Loading Model...')
    model = torch.load(f)
    breakpoint()

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(
            args.eval_tgt_len,  # tgt_len
            args.ext_len + args.tgt_len - args.eval_tgt_len,
            args.mem_len)  # mem_len
    else:
        model.reset_length(args.eval_tgt_len,
                           args.ext_len,
                           args.mem_len + args.tgt_len - args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems,
                        carry_over_fast_weight=args.carry_over_fast_weight)
#             import pdb;pdb.set_trace()
            breakpoint()
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len

va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

val_loss = evaluate(va_iter)

