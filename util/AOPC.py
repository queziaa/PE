
from copy import deepcopy
import torch

# 这段代码定义了一个名为calculate_metric的函数，它用于计算AOPC（Area Over the Perturbation Curve）指标。AOPC是一种用于评估模型解释性的指标，它通过测量模型预测概率随输入扰动变化的程度来评估模型的敏感性。

# 函数的输入参数包括：

# args：包含各种设置和参数的对象。
# predictor：用于进行预测的模型。
# label_id：目标标签的ID。
# probs：模型对原始输入的预测概率。
# tree：表示输入的树结构。
# aopc_token：表示扰动类型的字符串，可以是"del"（删除）或其他值（替换为填充符）。
# pad_token：用于替换的填充符。
# s_text：原始输入的文本。
# metric：要计算的指标，这里默认为"AOPC"。
# s_text_a和s_text_b：用于处理双输入任务（如自然语言推理）的额外输入。
# 在函数中，首先根据top和pool_size参数确定要进行扰动的单词数量k。然后，对每个要进行扰动的单词，根据aopc_token的值进行删除或替换操作，然后使用predictor计算扰动后的预测概率，并计算与原始预测概率的差值delta_p。所有的delta_p被添加到pool列表中。

# 最后，根据delta_p的值对pool列表进行排序，并返回前candidate_size个元素的delta_p值，这些值就是AOPC指标的结果。

# 总的来说，这个函数的主要作用是计算模型对输入扰动的敏感性，这是通过测量模型预测概率随输入扰动变化的程度来实现的。
def calculate_metric(args,predictor,label_id,probs,tree,aopc_token,pad_token,s_text,metric="AOPC",s_text_a=None,s_text_b=None):
    
    aopc_delta = []
    top = args.top
    if metric == "AOPC": 
        pool_size = args.pool_size
        k = max(int(len(tree) * top * pool_size), 1)
        # k = len(tree)
        pool = []
        # aopc_delta_per_example = []
        s_text_s = deepcopy(s_text)
        if s_text_a is not None:
            # print("s_text_a:{} ")
            s_text_a_ = deepcopy(s_text_a)
        if s_text_b is not None:
            s_text_b_ = deepcopy(s_text_b)
            
        print("#"*15)
        # if s_text_a is not None and s_text_b is not None:
        #     print("s_text_a:{} s_text_b:{}".format(s_text_a, s_text_b))
        for j in range(k):
            
            tok = tree[j]
            if aopc_token == "del":

                if tok in s_text_s:
                    s_text_s.remove(tok)
            
                mask = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s]
                # mask_ms = [0 if ll == pad_token  or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_ms]
                if s_text_a is not None:
                    if tok in s_text_a_:
                        s_text_a_.remove(tok)
                    elif tok in s_text_b_:
                        s_text_b_.remove(tok)
            else:
                for s_text_tok in range(len(s_text_s)):
                    
                    if s_text_s[s_text_tok] == tok:
                        s_text_s[s_text_tok] = pad_token
                mask = [0 if ll == pad_token or ll == "[PAD]" or ll == "<pad>" else 1 for ll in s_text_s ]

                if s_text_a is not None and s_text_b is not None:
                    f = True
                    for tid in range(len(s_text_a_)):
                        if s_text_a_[tid] == tok:
                            s_text_a_[tid] = pad_token
                            f = False
                            break
                    if f:
                        for tid in range(len(s_text_b_)):
                            if s_text_b_[tid] == tok:
                                s_text_b_[tid] = pad_token
                                break
                
                        # break
                        # cnt_tok1 -= 1
                
            
            with torch.no_grad():
                if args.dataset == "ptb" or args.dataset == "trec" or args.dataset == "yelp":
                    probs_i = predictor(input=s_text_s,mask = mask) 
                elif args.dataset == "snli":
                    probs_i = predictor(input = s_text_a_, mask = [1 if x_ != "[PAD]" else 0 for x_ in s_text_a_], input_else = s_text_b_, mask_else = [1 if x_ != "[PAD]" else 0 for x_ in s_text_b_])
                # print("tok:{} ")
                delta_p = probs[0,label_id].item()-probs_i[0,label_id].item()
                # print("tok:{} delta:{:.3f}".format(tok, delta_p))
                pool.append((tok,delta_p))
                # aopc_delta.append(delta_p)
                
            # print("text:{} tree size:{:d} tree edge1:{} tree edge2:{} p1:{:.4f} p2:{:.4f} AOPC:{:.4f}".format(s_text,len(tree),tok1,tok2,probs[0,label_id].item(),probs_i[0,label_id].item(),aopc_delta[-1]))
            # print("text:{}  tok:{} p1:{:.4f} p2:{:.4f} AOPC:{:.4f}".format(s_text,tok,probs[0,label_id].item(),probs_i[0,label_id].item(),aopc_delta[-1]))
    # return aopc_delta1,suff_delta1,logodd_delta1,flag_ms,flag_es
        
        pool = sorted(pool, key= lambda x : x[1], reverse=True)
        candidate_size = max(int(len(pool) / pool_size), 1)
        # for i,candidate_tok, score in enumerate(pool[:candidate_size]):
        #     print("dist_tok:{} tok:{} AOPC:{:.4f}".format(mst[i],candidate_tok,score))
        aopc_delta = [score for _,score in pool][:candidate_size]
        return aopc_delta
    else:
        pass