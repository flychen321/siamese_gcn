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
        feature = self.bn(feature)
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


class Sggnn(nn.Module):
    def __init__(self, basemodel=SiameseNet(ft_net_dense())):
        super(Sggnn, self).__init__()
        self.basemodel = basemodel
        self.rf = ReFineBlock(layer=2)
        self.fc = FcBlock()
        self.classifier = ClassBlock(input_dim=512, class_num=1)

    def forward(self, x):
        x_p = x[:, 0]
        x_g = x[:, 1:]
        len_feature = 1024
        d = torch.FloatTensor(len(x_p), len(x_g), len_feature).zero_()
        t = torch.FloatTensor(len(x_p), len(x_g), len_feature).zero_()
        w = torch.FloatTensor(len(x_g), len(x_g), len_feature).zero_()
        result = torch.FloatTensor(len(x_p), len(x_g)).zero_()
        for i in range(len(x_p)):
            for j in range(len(x_g)):
                d[i][j] = self.basemodel(x_p[i], x_g[j])[0]
                t[i][j] = self.rf(d[i][j])[0]
        for i in range(len(x_g)):
            for j in range(len(x_g)):
                w[i][j] = self.basemodel(x_g[i], x_g[j])[1]
        d_new = torch.mm(t, w)
        for i in range(len(x_p)):
            for j in range(len(x_g)):
                feature = self.fc(d_new[i][j])
                feature = self.classifier(feature)
                result[i][j] = feature
        return result


