import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from utils.log_uniform_sampler import LogUniformSampler, sample_logits
from utils.performer_helper import prime, draw_orthogonal_random_matrix
from utils.fast_weight import StepWiseLinearTransformerLayer
from utils.fast_weight import StepWiseDPFPLinearTransformerLayer
from utils.fast_weight import DebugStepWiseLinearTransformerLayer
# from utils.cuda_fast_weight_layer import CudaFastWeightLinearTransformerLayer
# from utils.cuda_fast_weight_layer import CudaFastWeightPerformerLayer
# from utils.cuda_fast_weight_layer import CudaFastWeightSumLinearTransformerLayer
# from utils.cuda_fast_weight_layer import CudaFastWeightSumPerformerLayer
# from utils.cuda_fast_weight_layer import CudaNormFastWeightLinearTransformerLayer
# from utils.cuda_fast_weight_layer import CudaNormFastWeightPerformerLayer
# from utils.cuda_fast_weight_layer import CudaFastWeightDPFPTransformerLayer
# from utils.cuda_fast_weight_layer import CudaNormFastWeightDPFPTransformerLayer

#from utils.fast_fast_weight import fast_weight_memory
#from utils.fast_transformers import fast_weight_sum
from utils.performer_helper import prime, draw_orthogonal_random_matrix
import time
import math

#import fourier_layer_cuda

def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output

# Fourier multihead attention.
# class FourierAttn(nn.Module):
#     def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, 
#                  pre_lnorm=False):
#         super(FourierAttn, self).__init__()

#         self.n_head = n_head
#         self.d_model = d_model
#         self.d_head = d_head
#         self.dropout = dropout

#         self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
#         self.k_net = nn.Linear(d_model, n_head * d_head, bias=False)
#         self.v_net = nn.Linear(d_model, n_head * d_head, bias=False)

#         self.drop = nn.Dropout(dropout)
#         self.dropatt = nn.Dropout(dropatt)
#         self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

#         self.layer_norm = nn.LayerNorm(d_model)

#         self.pre_lnorm = pre_lnorm
        
        
#         # self.paramR = nn.Parameter(2.0 * torch.ones(d_head), requires_grad= True) # 1 
#         self.paramR = nn.Parameter(2.0 * torch.ones(1), requires_grad= True) # 1 

#     def forward(self, h, attn_mask=None, mems=None,
#                 carry_over_fast_weight=False):
#         assert not carry_over_fast_weight, "Not supported."
#         # multihead attention
#         # [hlen x bsz x n_head x d_head]

#         if mems is not None:
#             c = torch.cat([mems, h], 0)
#         else:
#             c = h

#         if self.pre_lnorm:
#             # layer normalization
#             c = self.layer_norm(c)

#         head_q = self.q_net(h) # shape: [256, 48, 128] = [hlen, bsz, d_feature]
#         head_k = self.k_net(c)
#         head_v = self.v_net(c)

#         head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head) # [hlen, bsz, n_head, d_head] 
#         head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
#         head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

#         # [qlen x klen x bsz x n_head]
#         attn_score = FOURIERFunction.apply( head_q, head_k, self.paramR)
#         attn_prob = attn_score**4
                    
#         if attn_mask is not None and attn_mask.any().item():
#             if attn_mask.dim() == 2:
#                 attn_prob.masked_fill_(
#                     attn_mask[None,:,:,None], 0.0)
#             elif attn_mask.dim() == 3:
#                 attn_prob.masked_fill_(attn_mask[:,:,:,None], 0.0)

#         attn_prob = attn_prob / ((torch.sum(attn_prob, dim=1))[:, None, :, :] + 1e-6)
        
#         attn_prob = self.dropatt(attn_prob)
        
#         ########################################
        
#         # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
#         # -> [qlen x bsz x n_head x d_head]
#         attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
#         attn_vec = attn_vec.contiguous().view(
#             attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

#         # linear projection
#         attn_out = self.o_net(attn_vec)
#         attn_out = self.drop(attn_out)

#         if self.pre_lnorm:
#             # residual connection
#             output = h + attn_out
#         else:
#             # residual connection + layer normalization
#             output = self.layer_norm(h + attn_out)

#         return output

# class FOURIERFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, head_q, head_k, paramR):
#         #print('type(head_q): ', head_q.type());
#         #print('type(head_k): ', head_k.type());
#         #print('type(paramR): ', paramR.type());
#         output = fourier_layer_cuda.forward(head_q, head_k, paramR)
#         variables = head_q, head_k, paramR, output
#         #print('size(head_q) = ',head_q.size());
#         #print('size(head_k) = ',head_k.size());
#         #print('size(paramR) = ',paramR.size());
#         #print('size(output) = ',output.size());
#         ctx.save_for_backward(*variables)

#         return output

    # @staticmethod
    # def backward(ctx, grad):
    #     head_q, head_k, paramR, output = ctx.saved_tensors
    #     grads = fourier_layer_cuda.backward(
    #         grad.contiguous(), head_q, head_k, paramR, output)
    #     grad_head_q, grad_head_k, grad_p = grads
    #     return grad_head_q, grad_head_k, grad_p

# Standard multihead attention.
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, layer_num, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        # self.loss_func = None
        # if loss_func == 'huber'
        #     self.loss_func = self.phi_huber
        # elif loss_func == 'welsch':
        #     self.loss_func = self.phi
        # elif loss_func == 'welsch':
        #     self.loss_func = self.phi_geman_mcclure
        # else:
        #     print(f'loss_func not given or not recognized. Defaulting to {loss_func}...')
        
        #self.register_buffer('AvgGradEst', self.W, persistent=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt) # consider removing for gradient weight computation purposes
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)
        # self.smoothing_correction = smoothing_correction

        self.layer_num = layer_num
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False, head_q_last = None, head_v_last = None):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        # if self.over_layers:
        #     if isinstance(h, tuple):
        #         h, head_q_last, head_v_last = h # unpack tuple

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h # c, h: 256 x 48 x 128

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)
        
        head_q = self.q_net(h) #
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        if self.layer_num >= 14:
            attn_prob = attn_prob.permute(2, 3, 0, 1)
            attn_prob = 2 * attn_prob - attn_prob @ attn_prob
            attn_prob = attn_prob.permute(2, 3, 0, 1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)

        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

# Linear multihead attention from Katharopoulos et al.
# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
# https://arxiv.org/abs/2006.16236
class LinearMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(LinearMultiHeadAttn, self).__init__()
        print("Using LinearMultiHeadAttn --")
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
    

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)

        # Calculating attention scores
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # masked position to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# DPFP linear attention.
class DPFPMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, n_roll=3):
        super(DPFPMultiHeadAttn, self).__init__()
        print(f"Using DPFPMultiHeadAttn with {n_roll} rolls --")
        self.n_roll = n_roll
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def mul_roll_repeat(self, x):
        rolls = []
        for i in range(1, self.n_roll + 1):
            rolls.append(x * x.roll(shifts=i, dims=-1))
        return torch.cat(rolls, dim=-1)

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # transform q and k
        act = lambda x: F.relu(x)  # relu or exp
        head_k = torch.cat([act(head_k), act(-head_k)], dim=-1)
        head_q = torch.cat([act(head_q), act(-head_q)], dim=-1)

        head_k = self.mul_roll_repeat(head_k)
        head_q = self.mul_roll_repeat(head_q)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # masked position to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


# Performer multihead attention from Choromanski et al.
# Rethinking Attention with Performers. https://arxiv.org/abs/2009.14794
class PerformerMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, proj_dim=256, device='cuda',
                 skip_attn_normalization=False):
        assert not skip_attn_normalization, "Not implemented."
        # proj_dim: projected dimension
        print(f"Using PerformerMultiHeadAttn -- proj_dim: {proj_dim}")

        super(PerformerMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.proj_dim = proj_dim
        # so that we can keep the same matrix at test time.
        self.proj_matrix = draw_orthogonal_random_matrix(
            d_head, proj_dim, device=device)  # TODO store this as param?

    def forward(self, h, attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        assert not carry_over_fast_weight, "Not supported."
        # multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            # layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        if redraw:
            self.proj_matrix = draw_orthogonal_random_matrix(
                self.d_head, self.proj_dim, device=h.device)

        # transform q and k
        head_q = prime(head_q, self.proj_matrix)  # (len, B, n_head, proj_dim)
        head_k = prime(head_k, self.proj_matrix)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # set masked positions to 0
                attn_score.masked_fill_(attn_mask[None, :, :, None], 0)
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], 0)

        # normalize attn scores over keys
        eps = 1e-5
        denominator = torch.sum(attn_score, 1, keepdim=True) + eps
        # get (q_len, 1, B, n_head)

        attn_score = self.dropatt(attn_score)  # change
        attn_prob = attn_score / denominator

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(
            self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        # compute attention score
        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(
                    attn_score)

            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        # qlen x bsz x n_head x d_head
        rw_head_q = w_head_q + r_w_bias[None]

        # qlen x klen x bsz x n_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        # 1    x klen x 1   x n_head
        D_ = r_bias[None, :, None]
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(
                    attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class PerformerDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, attn_type,
                 **kwargs):
        super(PerformerDecoderLayer, self).__init__()
        # if attn_type == 25:
        #     attn_func = CudaFastWeightPerformerLayer
        # elif attn_type == 35:
        #     attn_func = CudaFastWeightSumPerformerLayer
        # elif attn_type == 45:
        #     attn_func = CudaNormFastWeightPerformerLayer
        if attn_type == 5:
            attn_func = PerformerMultiHeadAttn
        else:
            raise Exception(f"attn_type {attn_type} not allowed "
                            f"in PerformerDecoderLayer.")

        self.dec_attn = attn_func(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):

        output = self.dec_attn(
            dec_inp, attn_mask=dec_attn_mask, mems=mems, redraw=redraw,
            carry_over_fast_weight=carry_over_fast_weight)

        if carry_over_fast_weight:
            output, new_mem = output

        output = self.pos_ff(output)

        if carry_over_fast_weight:
            return output, new_mem

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, attn_type, layer_num,
                 **kwargs):
        super(DecoderLayer, self).__init__()
        print("ATTN_TYPE", attn_type)
        if attn_type == 2:
            attn_func = MultiHeadAttn
        elif attn_type == 4:
            attn_func = LinearMultiHeadAttn
        elif attn_type == 6:
            attn_func = DPFPMultiHeadAttn
        elif attn_type == 14:
            attn_func = StepWiseLinearTransformerLayer
        elif attn_type == 16:
            attn_func = StepWiseDPFPLinearTransformerLayer
        elif attn_type == 10:
            attn_func = DebugStepWiseLinearTransformerLayer
        # elif attn_type == 24:
        #     attn_func = CudaFastWeightLinearTransformerLayer
        # elif attn_type == 26:
        #     attn_func = CudaFastWeightDPFPTransformerLayer
        # elif attn_type == 44:
        #     attn_func = CudaNormFastWeightLinearTransformerLayer
        # elif attn_type == 46:
        #     attn_func = CudaNormFastWeightDPFPTransformerLayer
        # elif attn_type == 34:
        #     attn_func = CudaFastWeightSumLinearTransformerLayer
        # elif attn_type == 200:
        #     attn_func = MGKAttn
        # elif attn_type == 201:
        #     attn_func = MFAAttn
        # elif attn_type == 202:
        #     attn_func = GMMAttn
        # elif attn_type == 203:
        #     attn_func = FourierAttn
        # elif attn_type == 204:
        #     attn_func = PatchAttn
        # elif attn_type == 205:
        #     attn_func = HDPAttn
        else:
            raise Exception(f"attn_type {attn_type} not allowed here.")
        self.dec_attn = attn_func(n_head, d_model, d_head, dropout, layer_num, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))
        self.attn_type = attn_type

    def forward(self, *dec_inp, dec_attn_mask=None, mems=None, redraw=True,
                carry_over_fast_weight=False):
        
        output = self.dec_attn(*dec_inp, attn_mask=dec_attn_mask, mems=mems, carry_over_fast_weight=carry_over_fast_weight)
        
        if carry_over_fast_weight:
            output, new_mem = output

        # if self.over_layers:
        #     output, head_q_last, head_v_last = output

        output = self.pos_ff(output)

        if carry_over_fast_weight:
            return output, new_mem
        
        if False: #self.over_layers:
            return output, head_q_last, head_v_last
        else:
            return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias,
                dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None,
                mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)

        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, 
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], 
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(self,
                 n_token,
                 n_layer,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout,
                 dropatt,
                 M_positions = [],
                 show_M = False,
                 downsample_size = 0,
                 compare_downsample_grads = False,
                 smoothing_correction = 1,
                #  over_layers = False,
                 attenuation = 1e-3,
                 tie_weight=True,
                 d_embed=None,
                 div_val=1,
                 tie_projs=[False],
                 pre_lnorm=False,
                 tgt_len=None,
                 ext_len=None,
                 mem_len=None,
                 cutoffs=[],
                 adapt_inp=False,
                 same_length=False,
                 attn_type=0,
                 clamp_len=-1,
                 sample_softmax=-1,
                 proj_dim=256,  # for performer layers
                 n_roll=3,  # mirrored attention
                 skip_attn_normalization=False,
                 no_pos=False,  # no positional encoding
                 device='cuda',
                 update_mode='hard',
                 kernel_size=[1,1], 
                 stride=[1,1],
                 n_global_head=2):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.no_pos = no_pos

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(
            n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3, 4, 203]:  # absolute embeddings
            # 2: baseline vanilla transformer
            # 3:
            # 4: linear transformer
            for i in range(n_layer):
                if i in M_positions: # append an M-decoder layer 
                    self.layers.append(
                        DecoderLayer(
                            n_head, d_model, d_head, d_inner, dropout,
                            dropatt=dropatt, pre_lnorm=pre_lnorm,
                            attn_type=attn_type, M = True, show_M = show_M, downsample_size = downsample_size, 
                            compare_downsample_grads = compare_downsample_grads, smoothing_correction = smoothing_correction,
                            attenuation = attenuation, layer_num=i)
                    )
                else: # append a normal layer
                    self.layers.append(
                        DecoderLayer(
                            n_head, d_model, d_head, d_inner, dropout,
                            dropatt=dropatt, pre_lnorm=pre_lnorm,
                            attn_type=attn_type, layer_num=i)
                    )

        
        elif attn_type in [200, 201, 202]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, update_mode=update_mode)
                )
                
        elif attn_type in [204,]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout, kernel_size=kernel_size, stride=stride, 
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type)
                )
        
        elif attn_type in [205,]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout, n_global_head=n_global_head, 
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type)
                )
        
        elif attn_type in [6, 7]:  # absolute embeddings
            # 6: mirrored attention
            # 7: mirrored attention v2
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, n_roll=n_roll)
                )
        elif attn_type in [10, 14, 24, 34, 44]:  # fast weights
            # 10: debugging, same as linear trafo but step by step
            # 14: linear fast weight
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        skip_attn_normalization=skip_attn_normalization)
                )
        elif attn_type in [16, 26, 46]:  # fast weights w/ absolute embeddings
            # 10: debugging, same as linear trafo but step by step
            # 14: linear fast weight
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, layer_id=i, num_layer=n_layer,
                        n_roll=n_roll,
                        skip_attn_normalization=skip_attn_normalization)
                )
        elif attn_type in [5, 25, 35, 45]:  # absolute embeddings, performer
            # performer case needs to be separate from the case above
            # such that we can del with custom logic for random projections.
            for i in range(n_layer):
                self.layers.append(
                    PerformerDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm,
                        attn_type=attn_type, proj_dim=proj_dim, device=device,
                        skip_attn_normalization=skip_attn_normalization)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, 
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.attn_type == 0:  # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(
                torch.Tensor(self.n_head, self.d_head))

        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))

        elif self.attn_type in [2, 4, 5, 6, 7,
                                10, 14, 16,
                                24, 25, 26,
                                34, 35,
                                44, 45, 46, 200, 201, 202, 203, 204, 205]:
            # standard absolute pos
            self.pos_emb = PositionalEmbedding(self.d_model)

        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None, carry_over_fast_weight=False):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)
        if carry_over_fast_weight:
            mlen = 0
        else:
            mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                             + torch.tril(all_ones, -mask_shift_len)
                             ).bool()[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen),
                diagonal=1+mlen).bool()[:, :, None]

        hids = []
        if self.attn_type == 0:  # default
            pos_seq = torch.arange(
                klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)

            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)

            pos_emb = self.pos_emb(pos_seq)

            pos_emb = self.drop(pos_emb)
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type == 1:  # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len:]
                    r_bias = self.r_bias[i][-self.clamp_len:]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(
                    core_out, r_emb, self.r_w_bias[i], r_bias,
                    dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)

        elif self.attn_type in [2, 4, 5, 6, 7, 10, 14, 16, 200, 201, 202, 203, 204, 205]:  # absolute
            if self.no_pos:
                core_out = self.drop(word_emb)
            else:
                pos_seq = torch.arange(klen-1, -1, -1.0,
                                       device=word_emb.device,
                                       dtype=word_emb.dtype)
                # pos_seq = torch.arange(0, klen, device=word_emb.device,
                #                        dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)
                core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                if self.attn_type == 5:
                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i, redraw=self.training)
                else:
                    core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                     mems=mems_i)
                if isinstance(core_out, tuple):
                    hids.append(core_out[0]) # only append actual core_out, tuple of q,v_last
                else:
                    hids.append(core_out)

        elif self.attn_type in [24, 25, 26, 34, 35, 44, 45, 46]:  # absolute
            if self.no_pos:
                core_out = self.drop(word_emb)
            else:
                pos_seq = torch.arange(klen-1, -1, -1.0,
                                       device=word_emb.device,
                                       dtype=word_emb.dtype)
                # pos_seq = torch.arange(0, klen, device=word_emb.device,
                #                        dtype=word_emb.dtype)
                if self.clamp_len > 0:
                    pos_seq.clamp_(max=self.clamp_len)
                pos_emb = self.pos_emb(pos_seq)
                core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            if carry_over_fast_weight:
                new_mems = []
                
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if self.attn_type in [25, 35, 45]:
                    out = layer(
                        core_out, dec_attn_mask=dec_attn_mask,
                        mems=mems_i, redraw=self.training,
                        carry_over_fast_weight=carry_over_fast_weight)       
                else:
                    out = layer(
                        core_out, dec_attn_mask=dec_attn_mask, mems=mems_i,
                        carry_over_fast_weight=carry_over_fast_weight)
                
                if carry_over_fast_weight:
                    core_out, new_fast_weight = out
                    new_mems.append(new_fast_weight)
                else:
                    core_out = out
                hids.append(core_out)

        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(
                            mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(
                    core_out, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        
        if isinstance(core_out, tuple):
            core_out = core_out[0] # only take actual core_out at the final layer
        core_out = self.drop(core_out)
        if not carry_over_fast_weight:
            new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems, softmax_keep_order=False,
                carry_over_fast_weight=False):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.
        if not mems:
            mems = self.init_mems()

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(
            data, mems=mems, carry_over_fast_weight=carry_over_fast_weight)

        pred_hid = hidden[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)),
                             target.reshape(-1),
                             keep_order=softmax_keep_order)
            loss = loss.view(tgt_len, -1)
            
        # import pdb; pdb.set_trace()

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
        
    def get_pi(self):
        pi_list = []
        if self.attn_type == 200:
            for pi_indx in range(len(self.layers)):
                pi_list.append(self.layers[pi_indx].dec_attn.pi)
        return pi_list
    
    def get_pi0(self):
        pi_list = []
        if self.attn_type == 200:
            for pi_indx in range(len(self.layers)):
                pi_list.append(self.layers[pi_indx].dec_attn.pi0)
        return pi_list
    
    def get_pi1(self):
        pi_list = []
        if self.attn_type == 200:
            for pi_indx in range(len(self.layers)):
                pi_list.append(self.layers[pi_indx].dec_attn.pi1)
        return pi_list
    
    def get_pi0_data(self):
        pi_list = []
        if self.attn_type == 200:
            for pi_indx in range(len(self.layers)):
                pi_list.append(self.layers[pi_indx].dec_attn.pi0.data)
        return pi_list
    
    def get_pi1_data(self):
        pi_list = []
        if self.attn_type == 200:
            for pi_indx in range(len(self.layers)):
                pi_list.append(self.layers[pi_indx].dec_attn.pi1.data)
        return pi_list
    
    def get_mu_diff(self):
        md_list = []
        if self.attn_type == 200:
            for md_indx in range(len(self.layers)):
                md_list.append(-torch.sum((self.layers[md_indx].dec_attn.mu[0] - self.layers[md_indx].dec_attn.mu[1])**2))
        return md_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(
        data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(
                args.n_token, args.n_layer, args.n_head, args.d_model,
                args.d_head, args.d_inner, args.dropout, dropatt=args.dropout,
                tie_weight=True, d_embed=d_embed, div_val=div_val,
                tie_projs=tie_projs, pre_lnorm=True, tgt_len=tgt_len,
                ext_len=ext_len, mem_len=mem_len, cutoffs=cutoffs,
                attn_type=0, update_mode='hard').to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]

# SHARP WELSCH