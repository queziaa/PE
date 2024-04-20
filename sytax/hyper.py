import torch as th
from torch import nn
import geoopt as gt

from .probe import Probe
from .hyperrnn import HyperGRU


class PoincareProbeBase(Probe):
    def __init__(self, curvature: float, dim_hidden: int, **kwargs):
        # 初始化函数，设置曲率，隐藏层维度，并初始化一个Stereographic对象
        super().__init__(**kwargs)

        self.ball = gt.Stereographic(curvature)
        self.dim_hidden = dim_hidden

    def distance(self, transformed):
        """
        计算批次中每个句子的所有n^2对距离，距离是在指数映射后计算的。
        注意，由于填充，一些距离对填充项将是非零的。

        参数:
            transformed: 形状为(batch_size, max_seq_len, representation_dim)的单词表示批次
        返回:
            形状为(batch_size, max_seq_len, max_seq_len)的距离张量
        """
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        squared_distances = self.ball.dist2(transformed, transposed)
        return squared_distances

    def depth(self, transformed):
        """
        计算批次中每个句子的所有n个深度，深度是在指数映射后计算的。

        参数:
            transformed: 形状为(batch_size, max_seq_len, representation_dim)的单词表示批次
        返回:
            形状为(batch_size, max_seq_len)的深度张量
        """
        batchlen, seqlen, rank = transformed.size()
        norms = self.ball.dist0(transformed.reshape(batchlen * seqlen, 1, rank))
        norms = norms.reshape(batchlen, seqlen) ** 2
        return norms


class PoincareProbesytax(PoincareProbeBase):
    def __init__(self, **kwargs):
        # 初始化函数，首先调用父类的初始化方法，然后初始化两个参数proj和trans，这两个参数用于在投影操作中进行矩阵向量乘法
        super().__init__(**kwargs)

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def project(self, batch):
        # project方法是这个类的主要方法，它接收一个批次的单词表示，然后进行以下操作：
        # 1. 使用proj参数对批次进行矩阵乘法操作，得到transformed
        # 2. 使用指数映射将transformed映射到Poincaré球上
        # 3. 使用trans参数对transformed进行Möbius矩阵向量乘法操作
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed

class LocalPoincareProbe(PoincareProbeBase):
    """
    Computes squared poincare distance or depth by rnn with exponential map
    and a projection by Mobius mat-vec-mul.
    
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing LocalPoincareProbe")
        super().__init__(**kwargs)

        self.proj = nn.GRU(self.dim_in, self.dim_hidden, batch_first=True)
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.trans, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed, _ = self.proj(batch)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed


class LocalPoincareProbeWithHyperGRU(PoincareProbeBase):
    """
    Computes squared poincare distance or depth by rnn with exponential map
    and a projection by Mobius mat-vec-mul.
    
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing LocalPoincareProbeWithHyperGRU")
        super().__init__(**kwargs)

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.HyperGRU(
            self.dim_hidden,
            self.dim_out,
            ball=self.ball,
            default_dtype=self.default_dtype,
        )

        nn.init.uniform_(self.proj, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed, _ = self.trans(transformed)
        return transformed


class PoincareProbeNoSquare(Probe):
    """
    Poincare distance directly corresponds to tree distance
    no squared distance used
    """

    def __init__(self, curvature: float, dim_hidden: int, **kwargs):
        print("Constructing PoincareProbeNoSquare")
        super().__init__(**kwargs)

        self.ball = gt.Stereographic(curvature)
        self.dim_hidden = dim_hidden

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def distance(self, transformed):
        """
        Computes all n^2 pairs of distances after exponential map
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        squared_distances = self.ball.dist(transformed, transposed)
        return squared_distances

    def depth(self, transformed):
        """
        Computes all n depths after exponential map
        for each sentence in a batch.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of depths of shape (batch_size, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        norms = self.ball.dist0(transformed.reshape(batchlen * seqlen, 1, rank))
        norms = norms.reshape(batchlen, seqlen)
        return norms

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed
