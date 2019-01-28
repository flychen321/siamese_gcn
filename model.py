import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from scipy.io import loadmat
import os
import numpy as np
import math
import scipy.sparse as sp


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        num_ftrs = model_ft.fc.in_features  # extract feature parameters of fully collected layers
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs,
                                num_bottleneck)]  # add a linear layer, batchnorm layer, leakyrelu layer and dropout layer
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  # default dropout rate 0.5
        # transforms.CenterCrop(224),
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]  # class_num classification
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Fc_ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(Fc_ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class ReFineBlock(nn.Module):
    def __init__(self, input_dim=1024, dropout=True, relu=True, num_bottleneck=1024, layer=2):
        super(ReFineBlock, self).__init__()
        add_block = []
        for i in range(layer):
            add_block += [nn.Linear(input_dim, num_bottleneck)]
            add_block += [nn.BatchNorm1d(num_bottleneck)]
            if relu:
                add_block += [nn.LeakyReLU(0.1)]
            if dropout:
                add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class FcBlock(nn.Module):
    def __init__(self, input_dim=1024, dropout=True, relu=True, num_bottleneck=512):
        super(FcBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.fc = add_block

    def forward(self, x):
        x = self.fc(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim=512, class_num=751):
        super(ClassBlock, self).__init__()
        classifier = []
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class MaskBlock(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1024, kernel_size=1):
        super(MaskBlock, self).__init__()
        masker = []
        masker += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=1, padding=0, bias=True)]
        masker = nn.Sequential(*masker)
        masker.apply(weights_init_kaiming)
        self.masker = masker

    def forward(self, x):
        x = self.masker(x)
        return x


class ft_net_res(nn.Module):
    def __init__(self, class_num=6):
        super(ft_net_res, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = model_ft.fc.in_features  # extract feature parameters of fully collected layers
        model_ft.fc = FcBlock(input_dim=num_ftrs)
        self.model = model_ft
        self.classifier = ClassBlock()

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):
    def __init__(self, class_num=6):
        super(ft_net_dense, self).__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = FcBlock()
        self.model = model_ft
        self.classifier = ClassBlock()

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)
        # x = self.classifier(x)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.bn = nn.BatchNorm1d(1024)
        self.fc = FcBlock()
        self.classifier = ClassBlock(input_dim=512, class_num=1)

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)

        feature = (output1 - output2).pow(2)
        # feature = self.bn(feature)
        feature_fc = self.fc(feature)
        result = self.classifier(feature_fc)
        return feature, result

        # return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.FloatTensor(in_features, out_features)
        if bias:
            self.bias = torch.FloatTensor(out_features)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight.float())
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x)


class Sggnn_64_4images(nn.Module):
    def __init__(self, basemodel=SiameseNet(ft_net_dense())):
        super(Sggnn_64_4images, self).__init__()
        self.basemodel = basemodel
        self.rf = ReFineBlock(layer=2)
        # self.rf = FcBlock(input_dim=1024, dropout=True, relu=True, num_bottleneck=1024)
        self.fc = FcBlock()
        self.classifier = ClassBlock(input_dim=512, class_num=1)

    def forward(self, x, y):
        use_gpu = torch.cuda.is_available()
        x_p = x[:, :, 0]
        x_p = x_p.unsqueeze(2)
        x_g = x[:, :, 1:]
        x_p = x_p.contiguous().view(len(x_p), (len(x_p[0]) * len(x_p[0][0])), len(x_p[0][0][0]),
                                    len(x_p[0][0][0][0]), len(x_p[0][0][0][0][0]))
        x_g = x_g.contiguous().view(len(x_g), (len(x_g[0]) * len(x_g[0][0])), len(x_g[0][0][0]),
                                    len(x_g[0][0][0][0]), len(x_g[0][0][0][0][0]))

        num_p = len(x_p[0])  # 8
        num_g = len(x_g[0])  # 24
        num_g2 = np.square(num_g)  # 24*24 = 576
        batch_size = len(x_p)
        len_feature = 1024
        d = torch.FloatTensor(batch_size, num_p, num_g, len_feature).zero_()
        d_new = torch.FloatTensor(batch_size, num_p, num_g, len_feature).zero_()
        t = torch.FloatTensor(batch_size, num_p, num_g, len_feature).zero_()
        w = torch.FloatTensor(batch_size, num_g, num_g).zero_()
        result = torch.FloatTensor(batch_size, num_p, num_g).zero_()
        label = torch.LongTensor(batch_size, num_p, num_g).zero_()
        if use_gpu:
            d = d.cuda()
            d_new = d_new.cuda()
            t = t.cuda()
            w = w.cuda()
            result = result.cuda()
            label = label.cuda()

        y_p = y[:, :, 0]
        y_p = y_p.unsqueeze(2)
        y_g = y[:, :, 1:]
        y_p = y_p.contiguous().view(len(y_p), (len(y_p[0]) * len(y_p[0][0])))
        y_g = y_g.contiguous().view(len(y_g), (len(y_g[0]) * len(y_g[0][0])))

        print('batch_size = %d  num_p = %d  num_g = %d' % (batch_size, num_p, num_g))
        for i in range(num_p):
            for j in range(num_g):
                d[:, i, j] = self.basemodel(x_p[:, i], x_g[:, j])[0]
                t[:, i, j] = self.rf(d[:, i, j])
                if y_p[:, i] == y_g[:, j]:
                    label[:, i, j] = 1
                else:
                    label[:, i, j] = 0

        for i in range(num_g):
            for j in range(num_g):
                w[:, i, j] = self.basemodel(x_g[:, i], x_g[:, j])[1]

        # w need to be normalized
        for i in range(batch_size):
            w[i] = self.preprocess_adj(w[i])
        for i in range(t.shape[-1]):
            d_new[:, :, :, i] = torch.bmm(t[:, :, :, i], w)
        d_new = d

        for i in range(num_p):
            for j in range(num_g):
                feature = self.fc(d_new[:, i, j])
                feature = self.classifier(feature)
                result[:, i, j] = feature

        # label = torch.randint(0, 2, result.shape).cuda()

        print('run Sggnn foward success  !!!')
        return result, label

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj_np(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)


class Sggnn_end_to_end(nn.Module):
    def __init__(self, basemodel=SiameseNet(ft_net_dense())):
        super(Sggnn_end_to_end, self).__init__()
        self.basemodel = basemodel
        self.rf = ReFineBlock(layer=2)
        self.fc = FcBlock()
        self.classifier = ClassBlock(input_dim=512, class_num=1)

    def forward(self, x, y):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_p = x_p.unsqueeze(1)
        x_g = x[:, 1:]
        num_img = len(x[0])
        num_p_per_batch = len(x_p[0])  # 1
        num_g_per_batch = len(x_g[0])  # 3
        num_p = len(x_p) * len(x_p[0])  # 8
        num_g = len(x_g) * len(x_g[0])  # 24
        num_g2 = np.square(num_g)  # 24*24 = 576
        len_feature = 1024
        d = torch.FloatTensor(batch_size, batch_size, num_p_per_batch, num_g_per_batch, len_feature).zero_()
        t = torch.FloatTensor(batch_size, batch_size, num_p_per_batch, num_g_per_batch, len_feature).zero_()
        # this w for dynamic calculate the weight
        # w = torch.FloatTensor(batch_size, batch_size, num_g_per_batch, num_g_per_batch, 1).zero_()
        # this w for calculate the weight by label
        w = torch.FloatTensor(batch_size, batch_size, num_g_per_batch, num_g_per_batch).zero_()
        result = torch.FloatTensor(batch_size, batch_size, num_p_per_batch, num_g_per_batch).zero_()
        label = torch.LongTensor(batch_size, batch_size, num_p_per_batch, num_g_per_batch).zero_()
        if use_gpu:
            d = d.cuda()
            t = t.cuda()
            w = w.cuda()
            result = result.cuda()
            label = label.cuda()

        y_p = y[:, 0]
        y_p = y_p.unsqueeze(1)
        y_g = y[:, 1:]

        print('batch_size = %d  num_p = %d  num_g = %d' % (batch_size, num_p, num_g))
        for k in range(batch_size):
            x_g_temp1 = x_g[:k]
            x_g_temp2 = x_g[k:]
            x_g_temp = torch.cat((x_g_temp2, x_g_temp1), 0)
            y_temp1 = y_g[:k]
            y_temp2 = y_g[k:]
            y_temp = torch.cat((y_temp2, y_temp1), 0)

            for i in range(num_p_per_batch):
                for j in range(num_g_per_batch):
                    feature = self.basemodel(x_p[:, i], x_g_temp[:, j])[0]
                    t[k, :, i, j] = self.rf(feature)
                    label[k, :, i, j] = torch.where(y_p[:, i] == y_temp[:, j], torch.full_like(y_p[:, i], 1),
                                                    torch.full_like(y_p[:, i], 0))

            for i in range(num_g_per_batch):
                for j in range(num_g_per_batch):
                    # w[k, :, i, j] = self.basemodel(x_g[:, i], x_g_temp[:, j])[1]
                    w[k, :, i, j] = torch.where(y_g[:, i] == y_temp[:, j], torch.full_like(y_g[:, i], 1),
                                                torch.full_like(y_g[:, i], 0))

        d = d.reshape(batch_size, -1, len_feature)
        t = t.reshape(batch_size, -1, len_feature)
        w = w.reshape(batch_size * (num_img - 1), -1)
        label = label.reshape(batch_size, -1)
        result = result.reshape(batch_size, -1)

        # w need to be normalized
        w = self.preprocess_adj(w)
        for i in range(t.shape[-1]):
            d[:, :, i] = torch.mm(t[:, :, i], w)


        for i in range(num_p):
            # d_new[i, :] = self.rf(d_new[i, :])
            feature = self.fc(d[i, :])
            feature = self.classifier(feature)
            result[i, :] = feature.squeeze()

        # label = torch.randint(0, 2, result.shape).cuda()

        print('run Sggnn foward success  !!!')
        return result, label

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj_np(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)


class Sggnn_siamese(nn.Module):
    def __init__(self, siamesemodel):
        super(Sggnn_siamese, self).__init__()
        self.basemodel = siamesemodel


    def forward(self, x, y):
        use_gpu = torch.cuda.is_available()
        batch_size = len(x)
        x_p = x[:, 0]
        x_p = x_p.unsqueeze(1)
        x_g = x[:, 1:]
        num_img = len(x[0])
        num_p_per_id = len(x_p[0])  # 1
        num_g_per_id = len(x_g[0])  # 3
        num_p_per_batch = len(x_p) * len(x_p[0])  # 8
        num_g_per_batch = len(x_g) * len(x_g[0])  # 24
        len_feature = 1024
        d = torch.FloatTensor(batch_size, batch_size, num_p_per_id, num_g_per_id, len_feature).zero_()
        # this w for dynamic calculate the weight
        # w = torch.FloatTensor(batch_size, batch_size, num_g_per_id, num_g_per_id, 1).zero_()
        # this w for calculate the weight by label
        w = torch.FloatTensor(batch_size, batch_size, num_g_per_id, num_g_per_id).zero_()
        label = torch.LongTensor(batch_size, batch_size, num_p_per_id, num_g_per_id).zero_()

        if use_gpu:
            d = d.cuda()
            w = w.cuda()
            label = label.cuda()

        y_p = y[:, 0]
        y_p = y_p.unsqueeze(1)
        y_g = y[:, 1:]

        print('batch_size = %d  num_p_per_batch = %d  num_g_per_batch = %d' % (batch_size, num_p_per_batch, num_g_per_batch))
        for k in range(batch_size):
            x_g_temp1 = x_g[:k]
            x_g_temp2 = x_g[k:]
            x_g_temp = torch.cat((x_g_temp2, x_g_temp1), 0)
            y_temp1 = y_g[:k]
            y_temp2 = y_g[k:]
            y_temp = torch.cat((y_temp2, y_temp1), 0)

            for i in range(num_p_per_id):
                for j in range(num_g_per_id):
                    d[k, :, i, j] = self.basemodel(x_p[:, i], x_g_temp[:, j])[0]
                    label[k, :, i, j] = torch.where(y_p[:, i] == y_temp[:, j], torch.full_like(y_p[:, i], 1),
                                                    torch.full_like(y_p[:, i], 0))

            for i in range(num_g_per_id):
                for j in range(num_g_per_id):
                    # w[k, :, i, j] = self.basemodel(x_g[:, i], x_g_temp[:, j])[1]
                    w[k, :, i, j] = torch.where(y_g[:, i] == y_temp[:, j], torch.full_like(y_g[:, i], 1),
                                                torch.full_like(y_g[:, i], 0))

        # label = torch.randint(0, 2, result.shape).cuda()

        print('run Sggnn_siamese foward success  !!!')
        # return result, label
        return d, w, label

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj_np(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)


class Sggnn_gcn(nn.Module):
    def __init__(self):
        super(Sggnn_gcn, self).__init__()
        self.rf = ReFineBlock(layer=2)
        self.fc = FcBlock()
        self.classifier = ClassBlock(input_dim=512, class_num=1)

    def forward(self, d, w, label):
        use_gpu = torch.cuda.is_available()
        batch_size = len(d[0])
        num_p_per_id = len(d[0][0])  # 1
        num_g_per_id = len(d[0][0][0])  # 3
        num_p_per_batch = len(d[0]) * len(d[0][0])  # 1
        num_g_per_batch = len(d[0]) * len(d[0][0][0])  # 3
        len_feature = 1024
        t = torch.FloatTensor(d.shape).zero_()
        # this w for dynamic calculate the weight
        # w = torch.FloatTensor(batch_size, batch_size, num_g_per_batch, num_g_per_batch, 1).zero_()
        # this w for calculate the weight by label
        result = torch.FloatTensor(d.shape[: -1]).zero_()
        if use_gpu:
            d = d.cuda()
            t = t.cuda()
            w = w.cuda()
            result = result.cuda()
            label = label.cuda()

        print('batch_size = %d  num_p_per_batch = %d  num_g_per_batch = %d' % (batch_size, num_p_per_batch, num_g_per_batch))
        for k in range(batch_size):
            for i in range(num_p_per_id):
                for j in range(num_g_per_id):
                    t[k, :, i, j] = self.rf(d[k, :, i, j])

        d = d.reshape(batch_size, -1, len_feature)
        t = t.reshape(d.shape)
        w = w.reshape(batch_size * num_g_per_id, -1)
        label = label.reshape(batch_size, -1)
        result = result.reshape(batch_size, -1)

        # w need to be normalized
        w = self.preprocess_adj(w)
        for i in range(t.shape[-1]):
            d[:, :, i] = torch.mm(t[:, :, i], w)

        # maybe need to fix
        for i in range(num_p_per_batch):
            feature = self.fc(d[i, :])
            feature = self.classifier(feature)
            result[i, :] = feature.squeeze()

        # label = torch.randint(0, 2, result.shape).cuda()

        print('run Sggnn_gcn foward success  !!!')
        return result, label

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def preprocess_adj_np(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = adj + torch.eye(adj.shape[0]).cuda()
        rowsum = torch.Tensor(adj.sum(1).cpu()).cuda()
        d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
