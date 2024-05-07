import math

import torch as th
from torch import nn
import geoopt as gt


class EuclideanProbe(nn.Module):
    def __init__(
        self, device, default_dtype=th.float32, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.layer_num = layer_num

        self.probe_dim = 768
        self.bound = 1 / math.sqrt(self.probe_dim)
        self.pos = nn.Parameter(data=th.zeros(self.probe_dim))
        self.neg = nn.Parameter(data=th.zeros(self.probe_dim))
        nn.init.uniform_(self.pos, -self.bound, self.bound)
        nn.init.uniform_(self.neg, -self.bound, self.bound)

        self.proj = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        pos_logits = (((self.neg - transformed) ** 2).sum(-1)).sum(-1)
        neg_logits = (((self.pos - transformed) ** 2).sum(-1)).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)


class PoincareProbe(nn.Module):
    def __init__(
        self, device, default_dtype=th.float64, layer_num: int = 10, type = "ptb"
    ):
                # 初始化函数，首先调用父类的初始化方法，然后设置设备、数据类型、Poincaré球模型、探针维度、层级数和类型
        # 根据类型，初始化不同的中心点，并将它们映射到Poincaré球上
        # 初始化两个参数proj和trans，这两个参数用于在投影操作中进行矩阵向量乘法
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.ball = gt.Stereographic(-1)
        self.probe_dim = 64
        self.layer_num = layer_num
        self.type = type
        self.bound = 1 / math.sqrt(self.probe_dim)
        if type == "ptb" or type == "yelp":
            pos = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
            neg = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
            pos = self.ball.expmap0(pos)
            neg = self.ball.expmap0(neg)
            self.pos = gt.ManifoldParameter(data=pos, manifold=self.ball)
            self.neg = gt.ManifoldParameter(data=neg, manifold=self.ball)
        elif type == "snli":
            c1 = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
            c2 = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
            c3 = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
            c1 = self.ball.expmap0(c1)
            c2 = self.ball.expmap0(c2)
            c3 = self.ball.expmap0(c3)
            self.c1 = gt.ManifoldParameter(data=c1, manifold=self.ball)
            self.c2 = gt.ManifoldParameter(data=c2, manifold=self.ball)
            self.c3 = gt.ManifoldParameter(data=c3, manifold=self.ball)
        elif type == "trec":
            class_centriods = [th.zeros(self.probe_dim).uniform_(self.bound, self.bound)] * 6
            class_centriods = [self.ball.expmap0(c) for c in class_centriods]
            class_centriods = [gt.ManifoldParameter(data = c, manifold = self.ball) for c in class_centriods]
            self.centriods = nn.ParameterList(class_centriods)
            
        self.proj = nn.Parameter(data=th.zeros(768, self.probe_dim))
        self.trans = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def forward(self, sequence_output):
                # forward方法是这个类的主要方法，它接收一个序列输出，然后进行以下操作：
        # 1. 使用proj参数对序列输出进行矩阵乘法操作，得到transformed
        # 2. 使用指数映射将transformed映射到Poincaré球上
        # 3. 使用trans参数对transformed进行Möbius矩阵向量乘法操作
        # 4. 根据类型，计算transformed与不同中心点的距离，并返回这些距离
        # 检查序列输出的数据类型是否与proj参数的数据类型一致，如果不一致，则将序列输出的数据类型转换为proj参数的数据类型
        if sequence_output.dtype != self.proj.dtype:
            sequence_output=sequence_output.to(self.proj.dtype)

        # 使用proj参数对序列输出进行矩阵乘法操作，得到transformed
        transformed = th.matmul(sequence_output, self.proj)

        # 使用指数映射将transformed映射到Poincaré球上
        transformed = self.ball.expmap0(transformed)

        # 使用trans参数对transformed进行Möbius矩阵向量乘法操作
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        if self.type == "ptb" or self.type == "yelp":
            pos_logits = self.ball.dist(self.neg, transformed).sum(-1)
            neg_logits = self.ball.dist(self.pos, transformed).sum(-1)

            return th.stack((neg_logits, pos_logits), dim=-1)
        elif self.type == "snli":
            c1_logits = self.ball.dist(self.c1, transformed).sum(-1)
            c2_logits = self.ball.dist(self.c2, transformed).sum(-1)
            c3_logits = self.ball.dist(self.c3, transformed).sum(-1)
            return th.stack((c1_logits, c2_logits, c3_logits), dim = -1)
        elif self.type == "trec":
            # transformed = transformed.to(self.device)
            # print(transformed.device)
            dist = [self.ball.dist(c, transformed).sum(-1) for c in self.centriods]
            
            return th.cat(dist, dim = -1)
            

    def forward_logits(self,sequence_output):
                # forward_logits方法与forward方法类似，但它只返回transformed，不计算距离
        sequence_output=sequence_output.to(th.float32)
        # print(sequence_output.dtype,self.proj.dtype)
        transformed = th.matmul(sequence_output, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed


class EuclideanProbeFixed(nn.Module):
    def __init__(
        self, device, default_dtype=th.float32, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.layer_num = layer_num

        self.probe_dim = 768
        self.bound = 1 / math.sqrt(self.probe_dim)
        self.pos = th.ones(self.probe_dim).to(device) * self.bound
        self.neg = th.ones(self.probe_dim).to(device) * (-self.bound)

        self.proj = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        pos_logits = (((self.neg - transformed) ** 2).sum(-1)).sum(-1)
        neg_logits = (((self.pos - transformed) ** 2).sum(-1)).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)


class PoincareProbeFixed(nn.Module):
    def __init__(
        self, device, default_dtype=th.float64, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.ball = gt.Stereographic(-1)
        self.probe_dim = 64
        self.layer_num = layer_num

        self.bound = 0.5 / math.sqrt(self.probe_dim)
        self.pos = self.ball.expmap0(th.ones(self.probe_dim).to(device) * self.bound)
        self.neg = self.ball.expmap0(th.ones(self.probe_dim).to(device) * (-self.bound))

        self.proj = nn.Parameter(data=th.zeros(768, self.probe_dim))
        self.trans = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        pos_logits = self.ball.dist(self.neg, transformed).sum(-1)
        neg_logits = self.ball.dist(self.pos, transformed).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)
