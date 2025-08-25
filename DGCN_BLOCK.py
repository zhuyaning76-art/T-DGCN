import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from easytorch.device import to_device
from scipy.sparse.linalg import eigs

import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d


class T_cheby_conv_ds(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''

    def __init__(self, c_in, c_out, K, Kt):
        super(T_cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, Kt), padding=(0, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape

        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).repeat(nSample, 1, 1).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])

class SATT_3(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(SATT_3, self).__init__()
        self.conv1 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in * 12, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 4])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape
        seq = seq.permute(0, 1, 3, 2).contiguous().view(shape[0], shape[1] * 12, shape[3] // 12, shape[2])
        seq = seq.permute(0, 1, 3, 2)
        shape = seq.shape
        # b,c*12,n,l//12
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)
        # a,_ = torch.max(logits, -1, True)
        # logits = logits - a
        # logits = logits.permute(0,2,1,3).contiguous()
        # logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)
        return logits


class SATT_2(nn.Module):
    def __init__(self, c_in, num_nodes):
        super(SATT_2, self).__init__()
        self.conv1 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(c_in, c_in, kernel_size=(1, 1), padding=(0, 0),
                            stride=(1, 1), bias=False)
        self.bn = LayerNorm([num_nodes, num_nodes, 12])
        self.c_in = c_in

    def forward(self, seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 3, 1, 4, 2).contiguous()
        f2 = self.conv2(seq).view(shape[0], self.c_in // 4, 4, shape[2], shape[3]).permute(0, 1, 3, 4, 2).contiguous()

        logits = torch.einsum('bnclm,bcqlm->bnqlm', f1, f2)
        # a,_ = torch.max(logits, -1, True)
        # logits = logits - a
        # logits = logits.permute(0,2,1,3).contiguous()
        # logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits = logits.permute(0, 3, 1, 2, 4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits, -1)
        return logits


A = np.zeros((60, 60))
for i in range(12):
    for j in range(12):
        A[i, j] = 1
        A[i + 12, j + 12] = 1
        A[i + 24, j + 24] = 1
for i in range(24):
    for j in range(24):
        A[i + 36, j + 36] = 1
B = (-1e13) * (1 - A)
B = (torch.tensor(B)).type(torch.float32).cuda()


class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)

        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,n

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        ##normalization
        # logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
        #                                   bn_decay=bn_decay, scope='bn')
        # a,_ = torch.max(logits, 1, True)
        # logits = logits - a

        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits, -1)
        return coefs


class dgcn_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(dgcn_block, self).__init__()
        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size)
        self.SATT_3 = SATT_3(c_out, num_nodes)
        self.SATT_2 = SATT_2(c_out, num_nodes)
        self.dynamic_gcn = T_cheby_conv_ds(c_out, 2 * c_out, K, Kt)
        self.LSTM = nn.LSTM(num_nodes, num_nodes, batch_first=True)  # b*n,l,c
        self.K = K
        self.tem_size = tem_size
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
        # self.bn=BatchNorm2d(c_out)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = F.leaky_relu(x_1)
        x_tem1 = x_1[:, :, :, 0:12]
        x_tem2 = x_1[:, :, :, 12:24]
        S_coef1 = self.SATT_3(x_tem1)
        # print(S_coef1.shape)
        S_coef2 = self.SATT_2(x_tem2)
        # print(S_coef2.shape)
        S_coef = torch.cat((S_coef1, S_coef2), 1)  # b,l,n,c
        shape = S_coef.shape
        # print(S_coef.shape)
        h = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).cuda()
        c = Variable(torch.zeros((1, shape[0] * shape[2], shape[3]))).cuda()
        hidden = (h, c)
        S_coef = S_coef.permute(0, 2, 1, 3).contiguous().view(shape[0] * shape[2], shape[1], shape[3])
        S_coef = F.dropout(S_coef, 0.5, self.training)  # 2020/3/28/22:17
        _, hidden = self.LSTM(S_coef, hidden)
        adj_out = hidden[0].squeeze().view(shape[0], shape[2], shape[3]).contiguous()
        adj_out1 = (adj_out) * supports
        x_1 = F.dropout(x_1, 0.5, self.training)
        x_1 = self.dynamic_gcn(x_1, adj_out1)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        x_1 = F.dropout(x_1, 0.5, self.training)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, adj_out, T_coef


class DGCN(nn.Module):
    def __init__(
            self,
            supports,
            c_in=3,
            c_out=64,
            num_nodes=170,
            tem_size=24,
            K=3,
            out_dim=12,
            Kt=3):
        super(DGCN,self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.num_nodes = num_nodes
        self.supports = supports
        self.K = K
        self.Kt = Kt
        self.block1=dgcn_block(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=dgcn_block(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.bn=BatchNorm2d(c_in,affine=False)
        #self.bn=LayerNorm([c_out,num_nodes,tem_size])
        self.conv1=Conv2d(c_out,1,kernel_size=(1, 2),padding=(0,0),
                          stride=(1,2), bias=True)


        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)


    def forward(self,x_r,supports):

        x_r=self.bn(x_r)
        x=x_r
        adj = supports
        adj1 = np.array(adj)
        adj2 = adj1[0]
        adjs = scaled_Laplacian(adj2)
        supports1 = (torch.tensor(adjs)).type(torch.float32)
        supports1 = to_device(supports1)

        A=self.h+supports1
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
        #print(A1)
        x,_,_=self.block1(x,A1)
        #print(x.shape)
        x,d_adj,t_adj=self.block2(x,A1)
        #print(x.shape)

        x=self.conv1(x).squeeze()
        #print(x.shape)#b,n,l
        return x,d_adj,A
