import torch as th
from torch import nn
import geoopt as gt


class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, ball):
        super().__init__()

# HyperGRUCell是一个类，它实现了在超球面上的门控循环单元（GRU）的单元操作。GRU是一种常用的循环神经网络（RNN）结构，它通过引入“门”机制来控制信息的流动，以解决传统RNN在处理长序列时可能出现的梯度消失或梯度爆炸问题。

# 在传统的GRU中，所有的计算都在欧几里得空间中进行。而在HyperGRUCell中，所有的计算都在超球面上进行。超球面是一种非欧几里得空间，它可以帮助模型更好地处理复杂的数据结构，如图和树。

# HyperGRUCell类通常包含一个forward方法，该方法接收当前的输入和上一时刻的隐藏状态，然后计算出当前时刻的隐藏状态。这个过程包括计算更新门、重置门和新的候选隐藏状态，然后根据这些门的值来更新隐藏状态。

# 总的来说，HyperGRUCell类是用来实现在超球面上的GRU单元操作的，它是构建超球面上的GRU网络的基础。

        # 设置输入大小，隐藏层大小和超球面
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball

        # 计算权重初始化的范围
        k = (1 / hidden_size) ** 0.5

        # 初始化隐藏层的权重参数w_z, w_r, w_h，它们是在超球面上的
        self.w_z = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )
        self.w_r = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )
        self.w_h = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )

        # 初始化输入层的权重参数u_z, u_r, u_h，它们也是在超球面上的
        self.u_z = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )
        self.u_r = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )
        self.u_h = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )

        # 初始化偏置参数b_z, b_r, b_h，它们是在超球面上的，并且初始化为零
        self.b_z = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )
        self.b_r = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )
        self.b_h = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )

    def transition(self, W, h, U, x, hyp_b):
        # 使用Möbius矩阵向量乘法计算W otimes h和U otimes x
        W_otimes_h = self.ball.mobius_matvec(W, h)
        U_otimes_x = self.ball.mobius_matvec(U, x)

        # 使用Möbius加法计算W_otimes_h + U_otimes_x
        Wh_plus_Ux = self.ball.mobius_add(W_otimes_h, U_otimes_x)

        # 使用Möbius加法计算Wh_plus_Ux + hyp_b并返回结果
        return self.ball.mobius_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        # 计算门控信号z，使用sigmoid函数进行激活
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(self.ball.logmap0(z))

        # 计算门控信号r，使用sigmoid函数进行激活
        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(self.ball.logmap0(r))

        # 计算r与hidden的点乘，结果为r_point_h
        r_point_h = self.ball.mobius_pointwise_mul(hidden, r)

        # 计算新的隐藏状态h_tilde
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)

        # 计算-hidden与h_tilde的Möbius加法，结果为minus_h_oplus_htilde
        minus_h_oplus_htilde = self.ball.mobius_add(-hidden, h_tilde)

        # 计算新的隐藏状态new_h，它是hidden与minus_h_oplus_htilde与z的Möbius加法和点乘的结果
        new_h = self.ball.mobius_add(
            hidden, self.ball.mobius_pointwise_mul(minus_h_oplus_htilde, z)
        )

        # 返回新的隐藏状态new_h
        return new_h


class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ball, default_dtype=th.float64):
        # 初始化函数，设置输入大小，隐藏层大小，超球面和默认数据类型
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball
        self.default_dtype = default_dtype

        # 初始化一个HyperGRUCell对象，它是在超球面上的GRU单元
        self.gru_cell = HyperGRUCell(hidden_size, hidden_size, ball=self.ball)

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        # 初始化GRU的状态，它是一个全零的张量，形状为(batch_size, hidden_size)，数据类型为default_dtype，设备为cuda_device
        return th.zeros(
            (batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device
        )

    def forward(self, inputs):
        # forward函数是GRU的前向传播函数
        # 它首先初始化隐藏状态hidden
        # 然后对输入数据进行遍历，每次遍历都使用gru_cell更新hidden，并将hidden添加到outputs列表中
        # 最后，将outputs列表转换为张量，并进行转置，然后返回
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)
    
#     HyperGRU是一个类，它实现了在超球面上的门控循环单元（GRU）的网络结构。GRU是一种常用的循环神经网络（RNN）结构，它通过引入“门”机制来控制信息的流动，以解决传统RNN在处理长序列时可能出现的梯度消失或梯度爆炸问题。

# 在传统的GRU中，所有的计算都在欧几里得空间中进行。而在HyperGRU中，所有的计算都在超球面上进行。超球面是一种非欧几里得空间，它可以帮助模型更好地处理复杂的数据结构，如图和树。

# HyperGRU类通常包含一个forward方法，该方法接收一系列的输入，然后对每个输入进行处理，计算出对应的隐藏状态。这个过程是通过调用HyperGRUCell来实现的，HyperGRUCell是在超球面上的GRU单元。

# 总的来说，HyperGRU类是用来实现在超球面上的GRU网络的，它是构建超球面上的RNN模型的基础。