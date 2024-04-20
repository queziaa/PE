import torch as th

import numpy as np
from tqdm import tqdm

type_num = {"ptb": 2, "snli": 3, "trec": 6, "yelp" : 2}


def evaluate(test_data_loader, probe, bert, loss_fct):
    # 将模型设置为评估模式
    probe.eval()

    # 初始化损失和准确率
    loss = 0
    acc = 0

    # 对测试数据加载器中的每一个批次进行遍历
    for batch in tqdm(test_data_loader, desc="[Evaluate]"):
        # 从批次中获取输入数据和标签
        text_input_ids, text_attention_mask, text_token_type_ids, label = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
        )

        # 将输入数据和标签移动到模型所在的设备上
        text_input_ids, text_token_type_ids, text_attention_mask, label = (
            text_input_ids.to(probe.device),
            text_token_type_ids.to(probe.device),
            text_attention_mask.to(probe.device),
            label.to(probe.device),
        )

        # 不计算梯度，以节省内存并加速计算
        with th.no_grad():
            # 使用BERT模型对输入数据进行前向传播，得到隐藏状态
            outputs = bert(
                text_input_ids,
                attention_mask=text_attention_mask,
                token_type_ids=text_token_type_ids,
                output_hidden_states=True,
            )
            hidden_states = outputs[2]

            # 获取指定层的隐藏状态，并将其移动到模型所在的设备上
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )

            # 使用探针模型对隐藏状态进行前向传播，得到对数几率
            logits = probe(sequence_output)

        # 计算损失和准确率，并累加到总损失和总准确率上
        C = type_num["ptb"]
        l = loss_fct(logits.view(-1, C), label.view(-1))
        loss += l.item()
        acc += (logits.argmax(-1) == label).sum().item()

    # 返回平均损失和平均准确率
   