# coding=utf-8
"""
    @project: smallcap-main
    @Author：no-zjc
    @file： Net_Utils.py
    @date：2024/1/22 6:24
"""
import torch
from torch import nn
import numpy as np
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
# from transformers import AutoTokenizer


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


# 定义一个线性层，将 32 降为 1
class Liner_Down(nn.Module):
    def __init__(self, n):
        super(Liner_Down, self).__init__()
        # 定义线性层来处理第3维度
        self.linear = nn.Linear(n, 1)  # 将 n 映射为 1

    def forward(self, x):
        # 输入形状为 (n, 80, 32, 768)
        # 使用线性层处理第 3 维 (32)
        # 将 (n, 80, 32, 768) 转换为 (n, 80, 1, 768)
        x = self.linear(x.transpose(2, 3)).transpose(2, 3)
        # 输出形状变为 (n, 80, 1, 768)，现在移除维度1
        x = x.squeeze(2)  # 去掉第 3 维度 (1)
        # 输出的形状将是 (n, 80, 768)
        return x

class Generation_Tool():

    @staticmethod
    def get_max_no_hallucination(result_list, k=2):
        '''
        找打验证结果中幻觉概率最小的前k个结果的下标，并返回下标和数组的值
        '''

        # 初始化保存结果的列表
        top_k_list = []
        result_list_1 = np.array(result_list)

        # 使用快速选择算法找到第一个元素前 k 大的子数组的下标
        first_element_values = result_list_1[:, 0]  # 获取所有子数组的第一个元素
        top_k_indices = np.argpartition(-first_element_values, k-1)[:k]  # 使用快速选择算法找到前 k 大的元素的索引

        # 打印找到的前 k 个元素最大的子数组的下标
        for idx in top_k_indices:
            top_k_list.append(result_list_1[idx])

        return top_k_indices, top_k_list

    @staticmethod
    def get_word_type(word):
        # 进行词性标注
        tokens = word_tokenize(word)
        tags = pos_tag(tokens)

        # 判断词语是名词还是动词
        for tag in tags:
            if tag[1] in ['NN', 'NNS', 'NNP', 'NNPS']:
                return 1

            elif tag[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                return 2

            else:
                return 0

    @staticmethod
    def get_sg_candidate_words( visual_constraints, cur_word_label, top_k=100, prefix_words=None):
        # prefix_words_list = prefix_words.split(" ")


        if cur_word_label == 0:
            return []

        if cur_word_label == 1:
            obj_info_list = visual_constraints["object"]
            obj_list = []
            for item in obj_info_list:
                obj_list.append(item["object"])
            if len(obj_list) > top_k:
                obj_list = obj_list[:top_k]
            if obj_list is None:
                return []
            return obj_list

        if cur_word_label == 2:
            rel_info_list = visual_constraints["relation"]
            rel_list = []
            for item in rel_info_list:
                rel_list.append(item["relation"])
            rel_list = list(set(rel_list))
            if len(rel_list) > top_k:
                rel_list = rel_list[:top_k]
            if rel_list is None:
                return []

            return rel_list


    @staticmethod
    def _distance_calculation(self, word, lst):
        distance = 0
        for wrd in lst:
            try:
                distance += self.embedder.distance(word, str(wrd))
            except:
                print("distance compute error!")
        distance = distance / len(lst)
        return distance


    @staticmethod
    def get_sg_candidate_words_by_prefix_words(prefix_words, visual_constraints, cur_word_label, top_k=100):

        pass



    @staticmethod
    def get_cur_word_token_length(tokenizer, infer_token, cur_word):
        for i in range(len(infer_token)):
            last_word_token = infer_token[len(infer_token) - i - 1:]
            last_word = tokenizer.batch_decode([last_word_token])
            if last_word[0].replace(' ', '') == cur_word:
                cur_word_token_length = i + 1
                return cur_word_token_length


if __name__ == '__main__':
    from VideoCaption_Hallucination.File_Tools import File_Tools as FT
    sg_info = FT.load_json_data("/home/wy3/zjc_data/datasets/MSVD-QA/SG/20/zzit5b_-ukg_5_20" + "/scene_graph_info.json")
    print(Generation_Tool.get_sg_candidate_words(sg_info, 1, 100))