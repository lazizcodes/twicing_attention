import argparse
#import time
import math
import os, sys
#import itertools
#from datetime import datetime

#import numpy as np
import torch
#import torch.nn as nn
#import torch.optim as optim

from data_utils import get_lm_corpus
#from mem_transformer import MemTransformerLM
from utils.exp_utils import create_exp_dir
#from utils.data_parallel import BalancedDataParallel

print(f'torch version: {torch.__version__}')

parser = argparse.ArgumentParser(
    description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8'],
                    help='dataset name')
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
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)'
                    )
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)'
                    )
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10,
                    help='evaluation batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                     help='experiment directory.')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--performer_proj_dim', type=int, default=16,
                    help='projection dimension for performer layers.')
parser.add_argument('--dpfp_n_roll', type=int, default=2,
                    help='number of rolls for dpfp attention layers.')
parser.add_argument('--carry_over_fast_weight', action='store_true',
                    help='carry over fast weights.')
parser.add_argument('--skip_attn_normalization', action='store_true',
                    help='skip denominator in fast weights.')
parser.add_argument('--no_pos', action='store_true',
                    help='do not use positional encoding.')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument(
    '--dynamic-loss-scale', action='store_true',
    help='Use dynamic loss scaling.'
         'If supplied, this argument supersedes --static-loss-scale.')
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')
parser.add_argument('--update_mode', type=str, default='hard',
                    help='hard, soft, rbf')
parser.add_argument('--pi_reg', type=float, default=0.0,
                    help='weight of the pi norm regularizer')
parser.add_argument('--md_reg', type=float, default=0.0,
                    help='weight of distance between 2 keys regularizer')
parser.add_argument('--kernel_size', type=int, nargs='+', default=[1, 1],
                        help='kernel size of block matrices in PatchAttn.')
parser.add_argument('--stride', type=int, nargs='+', default=[1, 1],
                        help='stride of block matrices in PatchAttn.')

parser.add_argument('--n_global_head', type=int, default=2,
                    help='number of global head in HDP transformer.')

parser.add_argument('--M-positions', nargs = '+', type = int,
                    help='List of positions for M-attention')

parser.add_argument('--show-M', action = "store_true",
                    help='Show Mahalanobis transformation matrix.')
parser.add_argument('--downsample-size', type = int, default = -1,
                    help = 'Amount of downsampling to do in M computation, recommended to choose multiples \
                    of d_head. Must be less than sequence length. 0 for no downsampling i.e using all queries/keys')
parser.add_argument('--compare-downsample-grads', action = "store_true", help = 'compare downsampled average gradients to fully estimated gradients')

args = parser.parse_args()

logging = create_exp_dir(args.work_dir, scripts_to_save=None, debug=args.debug)


device = torch.device('cuda' if args.cuda else 'cpu')

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
para_model = model.to(device)

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
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len

corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

te_iter = corpus.get_iterator('test', args.eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

va_iter = corpus.get_iterator('valid', args.eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

# Run on test data.
logging('Evaluation of saved model...')

test_loss = evaluate(te_iter)
val_loss = evaluate(va_iter)
logging('=' * 100)

logging('| test loss {:5.2f} | test ppl {:9.3f}'.format(
        test_loss, math.exp(test_loss)))

logging('| val loss {:5.2f} | val ppl {:9.3f}'.format(
        val_loss, math.exp(val_loss)))


logging('=' * 100)