from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig, DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.distributions import Categorical
import torch.nn.functional as F
# from captum.attr import Saliency
class bertEncoder(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 从预训练模型的路径加载BERT配置，并设置类别数量
        self.bertconfig = BertConfig.from_pretrained(config.bert_path)
        self.bertconfig.num_labels = config.num_classes

        # 从预训练模型的路径加载BERT模型，并设置配置
        self.bertmodel = BertModel.from_pretrained(config.bert_path, config=self.bertconfig)

        # 设置类别数量
        self.num_labels = config.num_classes

        # 创建一个Dropout层，用于在训练过程中随机关闭一部分神经元，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建一个线性层，用于将BERT模型的输出转换为类别预测
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        # for name,param in self.bertmodel.named_parameters():
        #     if "embedding" in name:
        #         param.requires_grad = False
    # 定义前向传播方法，用于获取BERT模型的嵌入输出
    def forward_for_IG(self, input_ids, token_type_ids):
        # 使用BERT模型的嵌入层处理输入
        x = self.bertmodel.embeddings(input_ids, token_type_ids)
        return x

    # 定义前向传播方法，用于计算模型的输出和损失
    def forward(self, input_ids, attention_mask=None,return_pool=False, token_type_ids=None,  labels=None,
                position_ids=None, head_mask=None,output_hidden_states=False):
        # 使用BERT模型处理输入
        outputs = self.bertmodel(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask,output_hidden_states=output_hidden_states)
        # 获取BERT模型的输出
        pooled_emb = outputs[0]
        pooled_output = outputs[1]

        # 使用Dropout层和线性层处理BERT模型的输出
        pooled_output = self.dropout(pooled_output)
        logits_ = self.classifier(pooled_output)
        logits = F.softmax(logits_,dim=-1)

        # 如果提供了标签，则计算损失
        if labels is not None:
            if self.num_labels == 1:
                # 如果类别数量为1，则使用均方误差损失
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # 如果类别数量大于1，则使用交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, logits)

        # 根据参数返回不同的输出
        if not return_pool:
            if labels is not None:
                return outputs  # (loss), logits, (hidden_states), (attentions)
            elif not output_hidden_states:
                return logits
            else:
                return logits,outputs[2]
        else:
            return pooled_emb, pooled_output