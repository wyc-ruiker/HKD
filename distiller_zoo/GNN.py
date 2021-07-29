from __future__ import print_function

import torch
from torch import nn
import math
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from dgl.nn.pytorch.factory import KNNGraph
from dgl.nn.pytorch import TAGConv
import torch.nn.functional as F
import dgl.backend as B
import dgl.function as fn
import dgl
import numpy as np

eps = 1e-7
knn = 8

def cos_distance_softmax(x):
    soft = F.softmax(x, dim=2)
    w = soft.norm(p=2, dim=2, keepdim=True)
    return 1 - soft @ B.swapaxes(soft, -1, -2) / (w @ B.swapaxes(w, -1, -2)).clamp(min=eps)

def knn_graph(x, k):
    if B.ndim(x) == 2:
        x = B.unsqueeze(x, 0)
    n_samples, n_points, _ = B.shape(x)

    dist = cos_distance_softmax(x)

    fil = 1 - torch.eye(n_points, n_points)
    dist = dist * B.unsqueeze(fil, 0).cuda()
    dist = dist - B.unsqueeze(torch.eye(n_points, n_points), 0).cuda()

    k_indices = B.argtopk(dist, k, 2, descending=False)

    dst = B.copy_to(k_indices, B.cpu())
    src = B.zeros_like(dst) + B.reshape(B.arange(0, n_points), (1, -1, 1))

    per_sample_offset = B.reshape(B.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = B.reshape(dst, (-1,))
    src = B.reshape(src, (-1,))
    adj = sparse.csr_matrix((B.asnumpy(B.zeros_like(dst) + 1), (B.asnumpy(dst), B.asnumpy(src))))

    g = DGLGraph(adj, readonly=True)
    return g

class NCEAverage(nn.Module):
    def __init__(self, inputSize, outputSize, K):
        super(NCEAverage, self).__init__()
        self.K = K
        self.momentum = 0.9
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
    
    def update(self, l, ab, y):
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(self.momentum)
            l_pos.add_(torch.mul(l, 1 - self.momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)
            self.memory_l.index_copy_(0, y, updated_l)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(self.momentum)
            ab_pos.add_(torch.mul(ab, 1 - self.momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
            self.memory_ab.index_copy_(0, y, updated_ab)
    
    def get_smooth(self, l, ab, y):
        momentum = 0.75
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_l = l_pos.div(l_norm)

            ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(ab, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_ab = ab_pos.div(ab_norm)
        return updated_l.detach(), updated_ab.detach()

    def get_pos(self, y):
        l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
        ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
        return l_pos.detach(), ab_pos.detach()

    def forward(self, batchSize, y, idx=None):
        K = self.K

        weight_t = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
        weight_t = weight_t.view(batchSize, K, -1)

        weight_s = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
        weight_s = weight_s.view(batchSize, K, -1)

        return weight_t, weight_s

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x):
        bsz = x.shape[0]
        # label 指定每个采样的正样本 idx = 0
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss

class GNNLoss(nn.Module):
    def __init__(self, opt):
        super(GNNLoss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.gnn_s = Encoder(opt.feat_dim, opt.feat_dim)
        self.gnn_t = Encoder(opt.feat_dim, opt.feat_dim)
        self.contrast = NCEAverage(opt.feat_dim, opt.n_data, opt.nce_k).cuda()
        self.criterion = NCESoftmaxLoss()
        self.feat_size = opt.feat_dim

        u = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        v = torch.tensor([i for i in range(opt.batch_size * opt.nce_k)]).cuda()
        self.G_neg = dgl.graph((u, v)).to('cuda:0')

    def forward(self, epoch, f_s, l_s, f_t, l_t, idx, contrast_idx=None):
        batchSize = f_s.size(0)
        K = self.contrast.K
        T = 0.07

        weight_t, weight_s = self.contrast(batchSize, idx, contrast_idx)
        
        # graph indepandent
        f_es = self.embed_s(f_s)
        f_et = self.embed_t(f_t)
        f_us, f_ut = self.contrast.get_pos(idx)
        ls_pos = torch.einsum('nc,nc->n', [f_ut, f_es]).unsqueeze(-1)
        lt_pos = torch.einsum('nc,nc->n', [f_us, f_et]).unsqueeze(-1)

        ls_neg = torch.bmm(weight_t, f_es.view(batchSize, self.feat_size, 1)).squeeze()
        lt_neg = torch.bmm(weight_s, f_et.view(batchSize, self.feat_size, 1)).squeeze()

        out_s = torch.cat([ls_pos, ls_neg], dim=1)
        out_s = torch.div(out_s, T)
        out_s = out_s.contiguous()

        out_t = torch.cat([lt_pos, lt_neg], dim=1)
        out_t = torch.div(out_t, T)
        out_t = out_t.contiguous()
        
        loss = self.criterion(out_s) + self.criterion(out_t)

        if batchSize < knn:
            return loss
        
        # graph nn
        G_pos_s = knn_graph(l_s.detach(), knn)
        G_pos_s = G_pos_s.to('cuda:0')
        G_pos_s.ndata['h'] = f_es
        f_gs = self.gnn_s(G_pos_s)

        G_pos_t = knn_graph(l_t.detach(), knn)
        G_pos_t = G_pos_t.to('cuda:0')
        G_pos_t.ndata['h'] = f_et
        f_gt = self.gnn_t(G_pos_t)

        f_sgs, f_sgt = self.contrast.get_smooth(f_gs, f_gt, idx)

        gs_pos = torch.einsum('nc,nc->n', [f_sgt, f_gs]).unsqueeze(-1)
        gt_pos = torch.einsum('nc,nc->n', [f_sgs, f_gt]).unsqueeze(-1)

        gs_neg = torch.bmm(weight_t, f_gs.view(batchSize, self.feat_size, 1)).squeeze()
        gt_neg = torch.bmm(weight_s, f_gt.view(batchSize, self.feat_size, 1)).squeeze()

        out_gs = torch.cat([gs_pos, gs_neg], dim=1)
        out_gs = torch.div(out_gs, T)
        out_gs = out_gs.contiguous()

        out_gt = torch.cat([gt_pos, gt_neg], dim=1)
        out_gt = torch.div(out_gt, T)
        out_gt = out_gt.contiguous()
        
        loss_g = self.criterion(out_gs) + self.criterion(out_gt)
        
        self.contrast.update(f_es, f_et, idx)
        return loss + loss_g

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = TAGConv(in_dim, hidden_dim, k=1)
        self.l2norm = Normalize(2)

    def forward(self, g):
        h = g.ndata['h']
        h = self.l2norm(self.conv1(g, h))
        return h

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out