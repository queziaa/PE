import torch


def get_mst(alpha1,alpha2,alpha3,sim,depth,logits,batch_text):
     # get_mst函数用于获取最小生成树（Minimum Spanning Tree，MST）
    # 它接收三个权重参数alpha1、alpha2和alpha3，以及相似度sim、深度depth和对数几率logits
    # 如果logits不为None，则将sim、depth和logits进行加权求和
    # 否则，只对sim和depth进行加权求和
    # 然后，根据加权求和的结果对输入文本进行排序，返回排序后的文本
    if logits is not None:
        sim = [alpha1 * x1 + alpha2 * x2 + alpha3 * x3 for x1,x2,x3 in zip(sim,depth,logits)]
    else:
        sim = [alpha1 * x1 + alpha2 * x2 for x1,x2 in zip(sim,depth)]
    mst = [(tok,score) for tok,score in zip(batch_text,sim)]
    if alpha3 > 0:
        mst = sorted(mst, key = lambda x : x[1], reverse=True)
    else:
        mst = sorted(mst, key = lambda x : x[1])
    mst = [tok for tok,_ in mst if tok != "[PAD]" and tok != "[CLS]" and tok != "[SEP]"]
    return mst

class EdgeNode:    #没有没有没有没有没有没有没有没有没有被使用
        # EdgeNode类用于表示边和节点
    # 它包含两个节点x和y，以及一个值v
    # 它还定义了一个__lt__方法，用于比较两个EdgeNode对象的v值
    def __init__(self,x,y,v) -> None:
        self.x = x
        self.y = y
        self.v = v

    def __lt__(self, other):
        return other.v > self.v

@torch.no_grad()
def calculate_level(model, level_list, input, mask):
     # calculate_level函数用于计算模型的层级
    # 它接收一个模型model、一个层级列表level_list、一个输入input和一个掩码mask
    # 它首先创建一个单位矩阵，并将其转换为掩码mask_
    # 然后，对于level_list中的每个层级，将mask_中对应的层级设置为0
    # 接着，将mask和mask_进行元素乘法，得到新的mask
    # 然后，将input和mask进行元素乘法，得到变量张量variant_tensor
    # 最后，使用模型model计算variant_tensor的对数几率logits，并返回logits

    # input : 1, L, d
    # mask : 1, L, L
    # pred : func
    # level_list : List[int]
    n = input.size(1)
    identity_matrix = torch.eye(n).reshape(1, n, n)
    mask_ = 1 - identity_matrix
    for level in level_list:
        mask_[:, :, level] = 0
    mask = mask * mask_
    variant_tensor = input * mask # L,L,d
    mask = mask.expand(n, n,-1) # L,L,L
    logits = model(variant_tensor, mask) # 
    return logits # L,C