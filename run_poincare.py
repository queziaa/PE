import os
import sys
import pickle
import math
import time
from argparse import ArgumentParser

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils as tutils
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
# from transformers.configuration_bert import BertConfig
import geoopt as gt

import numpy as np
from tqdm import tqdm

from probe.probe import *
from util.train import train
from util.evalu import evaluate
from util.cuda import get_max_available_gpu
from encoder.bert_encoder1 import *

default_dtype = th.float64
th.set_default_dtype(default_dtype)

log_path = os.path.join("./log")

_layer_num = 11
_run_num = 5
_epoch_num = 40
_batch_size = 32
_stop_lr = 5e-8

if __name__ == "__main__":
    """
    config
    """
    argp = ArgumentParser()  # 创建一个命令行参数解析器

    # 添加各种命令行参数
    argp.add_argument("--save", type=bool, default=True, help="Save probe")  # 是否保存探针
    argp.add_argument("--cuda", type=int, help="CUDA device")  # CUDA设备编号
    argp.add_argument("--bert_path",type=str,default="")  # BERT模型的路径
    argp.add_argument("--num_classes",default=2,type=int,help="number of classes")  # 类别数量
    argp.add_argument("--hidden_dropout_prob",type=float,default=0.1)  # 隐藏层的dropout概率
    argp.add_argument("--hidden_size",type=int,default=768)  # 隐藏层的大小
    argp.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")  # Adam优化器的初始学习率
    argp.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")  # 权重衰减系数
    argp.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # Adam优化器的epsilon值
    argp.add_argument("--poincare_type",type=str, default="snli")  # Poincare模型的类型

    args = argp.parse_args()  # 解析命令行参数
    bert_pretrained_file = args.bert_path  # 获取BERT模型的路径
    # 如果命令行参数中指定了CUDA设备，则使用指定的设备；否则，获取可用GPU最大的设备ID
    if args.cuda is not None:
        device_id = args.cuda
    else:
        device_id, _ = get_max_available_gpu()

    # 如果CUDA可用，则使用GPU，否则使用CPU
    device = "cuda:0"
    if th.cuda.is_available():
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")

    # 获取当前时间字符串
    timestr = time.strftime("%m%d-%H%M%S")

    # 如果日志路径不存在，则创建该路径
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 设置数据路径
    data_path = f"./data/{args.poincare_type}"

    # 加载训练、验证和测试数据集
    train_dataset = th.load(os.path.join(data_path, "train_dataset.pt"))
    dev_dataset = th.load(os.path.join(data_path, "dev_dataset.pt"))
    test_dataset = th.load(os.path.join(data_path, "test_dataset.pt"))

    # 创建数据加载器，用于在训练和测试过程中加载数据
    train_data_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
    dev_data_loader = DataLoader(dev_dataset, batch_size=_batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

    # 使用预训练的BERT模型文件初始化BERT模型
    bert = BertModel.from_pretrained(bert_pretrained_file)

    # 遍历BERT模型的所有参数，并将它们设置为不需要梯度，这意味着在训练过程中，这些参数不会被更新，我们不对BERT模型进行微调
    for param in bert.parameters():
        param.requires_grad = False

    # 将BERT模型移动到指定的设备上，如果有GPU可用，则使用GPU，否则使用CPU
    bert.to(device)

    # 设置日志文件的路径，文件名包含了层级数和当前时间
    log_file = os.path.join(
        log_path, "layer-" + str(_layer_num) + "-" + timestr + ".log"
    )

    # 初始化一个空列表，用于存储平均准确率
# 初始化一个空列表，用于存储平均准确率
    avg_acc = []

    # 对于每一次运行
    for run in tqdm(range(_run_num), desc="[Run]"):
        # 初始化PoincareProbe模型，并将其移动到指定的设备上
        probe = PoincareProbe(
            device=device, default_dtype=default_dtype, layer_num=_layer_num, type = args.poincare_type
        )
        probe.to(device)

        # 设置损失函数为交叉熵损失
        loss_fct = nn.CrossEntropyLoss()

        # 根据Poincare模型的类型，设置不同的优化器参数
#         args.poincare_type可能代表的是使用的Poincaré模型的类型。Poincaré模型是一种用于表示和处理层次结构和树形结构的模型，它可以捕获数据的层次性和相似性。

# 例如，如果你的任务是处理自然语言处理中的句法分析问题，你可能会有一个"ptb"（Penn Treebank）类型的Poincaré模型。如果你的任务是处理文本分类问题，你可能会有一个"trec"（Text REtrieval Conference）类型的Poincaré模型。如果你的任务是处理自然语言推理问题，你可能会有一个"snli"（Stanford Natural Language Inference）类型的Poincaré模型。
        if args.poincare_type == "ptb" or args.poincare_type == "yelp":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.pos},
                    {"params": probe.neg},
                ],
                lr=1e-3,
            )
        elif args.poincare_type == "snli":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.c1},
                    {"params": probe.c2},
                    {"params": probe.c3}
                ],
                lr = 1e-3
            )
        elif args.poincare_type == "trec":
            optimizer = gt.optim.RiemannianAdam(
                [
                    {"params": probe.proj},
                    {"params": probe.trans},
                    {"params": probe.centriods[0]},
                    {"params": probe.centriods[1]},
                    {"params": probe.centriods[2]},
                    {"params": probe.centriods[3]},
                    {"params": probe.centriods[4]},
                    {"params": probe.centriods[5]},
                ],
                lr = 1e-3
            )

        # 设置学习率调度器，当验证损失不再下降时，将学习率乘以0.1
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=0)

        # with open(log_file, "a") as f:
        #     f.write(f"Run: {run + 1}\n")
        print(f"Run: {run + 1}\n")
        for epoch in tqdm(range(_epoch_num), desc="[Epoch]"):

            start_time = time.time()
            train_loss, train_acc, dev_loss, dev_acc = train(
                train_data_loader,
                probe,
                bert,
                loss_fct,
                optimizer,
                dev_data_loader=dev_data_loader,
                scheduler=scheduler,
            )

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            # 如果学习率低于预设的停止学习率，或者已经达到最大训练周期，则进行测试
            if optimizer.param_groups[0]["lr"] < _stop_lr or epoch == _epoch_num - 1:
                # 调用evaluate函数进行测试，返回测试损失和测试准确率
                test_loss, test_acc = evaluate(test_data_loader, probe, bert, loss_fct)
            #    train_loss：这是在训练集上的损失。模型在训练过程中试图最小化这个损失。这个损失值可以帮助我们了解模型在训练数据上的表现。

            #     dev_loss（也被称为验证损失）：这是在验证集上的损失。验证集用于在训练过程中调整模型的参数（如学习率）和检查模型的过拟合情况。如果模型在训练集上表现良好，但在验证集上表现较差，那么可能存在过拟合问题。

            #     test_loss：这是在测试集上的损失。测试集用于在模型训练完成后评估模型的性能。这个损失值可以帮助我们了解模型在未见过的数据上的表现。

            #     在这段代码中，train_loss、dev_loss和test_loss都是通过调用损失函数loss_fct计算得到的。这三个损失值可以帮助我们了解模型在不同数据集上的表现，并据此调整模型的参数和结构。
                
                print(
                    f"Epoch: {epoch + 1} | time in {mins:.0f} minutes, {secs:.0f} seconds\n"
                )
                print(
                    f"\tTrain Loss: {train_loss:.4f}\t|\tTrain Acc: {train_acc * 100:.2f}%\n"
                )
                print(
                    f"\tDev Loss: {dev_loss:.4f}\t|\tDev Acc: {dev_acc * 100:.2f}%\n"
                )
                print(
                    f"\tTest Loss:  {test_loss:.4f}\t|\tTest Acc:  {test_acc * 100:.2f}%\n"
                )
                print("-" * 50 + "\n")

                break

        avg_acc.append(test_acc)
        # if args.save:
        # if 
        probe_ckeckpoint = f"./checkpoint/{args.poincare_type}/layer=11_dim=64.pt"
        th.save(probe.state_dict(), probe_ckeckpoint)

    # with open(log_file, "a") as f:
    #     f.write(f"Avg Acc: {np.mean(avg_acc)*100:.2f}%\n")
    print(f"Avg Acc: {np.mean(avg_acc)*100:.2f}%\n")
