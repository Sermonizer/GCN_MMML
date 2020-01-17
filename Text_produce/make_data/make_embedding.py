import numpy as np
import gensim
import pandas as pd
import json
import sys
import torch
# sys.path.append()

# a = [['1','2','3','4'],['1','3','4','5']]
# dick = {}
# for i in range(2):
#     for j in a[i]:
#         dick.setdefault(i, []).append(j)
        # dick.add(j)
# print(dick)
# print(torch.Tensor(a))

glove_file = '/home/administer/Tengxu/DataSet/Glove/glove.6B/glove.6B.300d.txt'

# def get_word_embedded(word2id, word_list, graph_list):

def get_word_embed(path):
    word_embed = {}
    with open(path, 'r') as file:
        # 存放词典的词的embedding
        embed = []
        # 存放词典中的词
        word = []
        num = 0
        for line in file.readlines():
            row = line.split()
            word.append(row[0])
            embed.append(row[1:])
            # for i in embed[num]:
            # # embed = [float(nums) for nums in embed]
            # #     word_embed.setdefault(word[num], []).append(i)
            #     word_embed[word[num]] = i
            embed_float = map(float, embed[num])
            word_embed[word[num]] = list(embed_float)
            num += 1
    return word_embed

get_word_embed(glove_file)
