import smart_open
from gensim import logger
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.clip_grad import clip_grad_norm

sys.path.append('/home/administer/Tengxu/Code/Text_produce/SceneGraphParser')
import sng_parser

glove_file = '/home/administer/Tengxu/DataSet/Glove/glove.6B/glove.6B.300d.txt'
tmp_file = '/home/administer/Tengxu/DataSet/Glove/glove.6B/glove2word_vec.txt'
flickr30k = '/media/administer/新加卷/DataSet/Multi modal/Flickr30k/flickr30k/dataset.json'
# glove2word2vec(glove_file, tmp_file)
# model = KeyedVectors.load_word2vec_format(tmp_file)
# model.save('/home/administer/Tengxu/Code/Modal/word2vec_model.model')
# model.save_word2vec_format('/home/administer/Tengxu/Code/Modal/word2vec_model.txt', binary=False)
# model.save_word2vec_format('/home/administer/Tengxu/Code/Modal/word2vec_model.bin.gz', binary=True)
model = KeyedVectors.load('/home/administer/Tengxu/Code/Modal/word2vec_model.model')

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# 获取句子、短语的列表、词典映射
def make_word(path):
    with open(path) as f:
        line = f.readline()
        sentence = json.loads(line)
        # 存放词在词典中的位置
        word2id = {}
        # 存放每个句子的词在词典中的位置
        word_list = []
        # 存放句子的每个短句的词在词典中的位置
        graph_list = []
        # 存放短句
        dicks = []
        num = 0
        # 词典中不存在的数量
        over_count = 0
        for i in sentence['images']:
            count = 0
            for j in range(5):
                word_list.append([])
                # 分词后的词
                tokens = i['sentences'][j]['tokens']
                # 原始句子
                raw = i['sentences'][j]['raw']
                for word in tokens:
                    # 约出现1900000次有效词，2000次左右无效词
                    if word in model.vocab:
                        word2id[word] = model.vocab[word].index
                        word_list[count].append(model.vocab[word].index)
                    else:
                        word2id[word] = 400000 + over_count
                        word_list[count].append(-1)
                        over_count += 1
                count += 1
                # 句法分析图
                graph = sng_parser.parse(raw)
                dicks.append([])
                # print(dicks[num])
                graph_list.append([])
                for k in graph['entities']:
                    dicks[num].append(str.lower(k['span']))
                    # print(dicks[num])
                times = 0
                for l in dicks[num]:
                    # print(len(dicks[num]))
                    # print(graph_list)
                    # print(dicks[num], l)
                    graph_list[num].append([])
                    for s in l.split():
                        if s in model.vocab:
                            graph_list[num][times].append(model.vocab[s].index)
                        else:
                            graph_list[num][times].append(-1)
                    times += 1
                num += 1
    return word2id, word_list, graph_list

# 从下载的Glove词向量中获取Embedding
def get_word_embedded(word2id, word_list, graph_list):
    word_embed = {}
    with open(glove_file, 'r') as file:
        # 存放词典的词的embedding
        embed = []
        # 存放词典中的词
        word = []
        num = 0
        for line in file.readlines():
            row = line.split()
            word.append(row[0])
            embed.append(row[1:])
            # embed = [float(nums) for nums in embed]
            # for i in range(len(embed)):
            #     word_embed[word] = embed[i]
            embed_float = map(float, embed[num])
            word_embed[word[num]] = list(embed_float)
            num += 1
            # print(embed)

    id2word = {id: w for w, id in word2id.items()}
    id2embed = {}
    for id in range(len(word2id)):
        if id2word[id] in word_embed:
            id2embed[id] = word_embed[id2word[id]]
        else:
            id2embed[id] = np.random.uniform(-0.25, 0.25, 300)
    # word2id --- id2embed 获取单词的嵌入
    word_embedding = [id2embed[id] for id in range(len(word2id))]
    return word_embedding

class EncodeTxt(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, hidden_size, use_bi_gru=False, no_txtnorm=False):
        super(EncodeTxt, self).__init__()
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # Embedding层的输出是： [seq_len,batch_size,embedding_size]
        self.embed = nn.Embedding(vocab_size, embed_size)

        self.use_bi_gru = use_bi_gru
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        # 初始化一下词向量
        self.init_weights()

    # 词向量初始化
    def init_weights(self):
            self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, length):
        # 处理不同长度的句子, length就是句子的长度list
        x = self.embed(x)
        packed = pack_padded_sequence(x, length, batch_first=True)

        # RNN的前向传播
        # out：PackedSequence对象
        out, hidden = self.gru(packed)

        # reshape最后的输出形状为：batch_size, hidden_size
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] + cap_emb[:, :, cap_emb.size(2) / 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len

# 超参数的设定
embed_size = 300    # 词嵌入的维度
hidden_size = 1024  # 使用RNN变种LSTM单元   LSTM的hidden size
num_layers = 1      #循环单元/LSTM单元的层数
num_epochs = 5      # 迭代轮次
num_samples = 1000  # 测试语言模型生成句子时的样本数
batch_size = 20     # 一批样本的数量
seq_length = 30     # 一个样本/序列长度
learning_rate = 0.002   # 学习率


# 有GPU用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word2id, word_list, graph_list = make_word(flickr30k)
word_embedding = get_word_embedded(word2id, word_list, graph_list)
txt_enc = EncodeTxt(len(word2id), embed_size, num_layers, hidden_size).to(device)

# 使用Adam优化方法 最小化损失 优化更新模型参数
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# for epoch in range(num_epochs):
#     for i in range(0, )
cap_emb = txt_enc.forward()
print(cap_emb)


# 整体模型训练，暂时不做
class train_TXT(object):
    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False, no_txtnorm=False):
        self.txt_enc = EncodeTxt(vocab_size, word_dim, embed_size, num_layers, use_bi_gru, no_txtnorm)

        if torch.cuda.is_available():
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # 交叉熵损失
        self.criterion = nn.CrossEntropyLoss()

        params = list(self.txt_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

    def state_dict(self):
        state_dict = [self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.txt_enc.load_state_dict(state_dict)

    def train_start(self):
        self.txt_enc.train()

    def forward_emb(self, captions, lengths, volatile=False):
        # 计算文本 embeddings
        # Set mini-batch dataset
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            captions = captions.cuda()

        # Forward
        # cap_embed (tensor), cap_lens (list)
        cap_embed, cap_lens = self.txt_enc(captions, lengths)
        return cap_embed, cap_lens

    def forward_los(self, cap_embed, cap_len):
        loss = self.criterion(cap_embed, cap_len)
        # self.logger.update('Le', )
        return loss

    def train_emb(self, captions, lengths, ids=None, *args):
        # 训练阶段
        # self.Eiters += 1
        # self.logger.update('Eit', self.Eiters)
        # self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        cap_embed, cap_lens = self.forward_emb(captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(cap_emb, cap_lens)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

# def get_word2id(path):
#     word2id = {}
#     with open(path, 'r') as f:
#         line = f.readline()
#         sentence = json.loads(line)
#         id = 0
#         for i in sentence['image']:
#             for j in range(5):
#                 tokens = i['sentences'][j]['tokens']
#                 for word in tokens:
#                     if word in model.vocab:
#                         word2id[id] = model.vocab[word].index
#                     else:
#                         word2id[id] = -1
#                     id += 1
#     return word2id

# get_word2id(glove_file)

# class Dictionary(object):
#     '''
#     构建词典
#     '''
#     def __init__(self):
#         self.word2id = {}   # 词到索引的映射
#         self.id2word = {}   # 索引到词的映射
#         self.id = 0
#
#     def add_word(self, word):
#         if not word in self.word2id:
#             self.word2id[word] = self.id
#             self.id2word[self.id] = word
#             self.id += 1
#
#     def __len__(self):
#         return len(self.word2id)
#
# class Corpus(object):
#     '''
#     基于训练语料，构建词典
#     '''
#     def __init__(self):
#         self.dictionary = Dictionary()
#
#     def get_data(self, path, batch_size=20):
#         with open(path, 'r') as file:
#             tokens = 0
#             for line in file.readlines():
#                 row = line.strip().split()
#                 words = row[0]
#                 tokens += len(words)
#             for word in words:
#                 self.dictionary.add_word(word)
#
#         id = torch.LongTensor(tokens)
#         token = 0
#         with open(path, 'r') as file:
#             for line in file:
#                 row = line.strip().split()
#                 words = row[0]
#                 for word in words:
#                     id[token] = self.dictionary.word2id[word]
#                     token += 1
#         num_batches = id.size(0) // batch_size
#         id = id[:num_batches * batch_size]
#         return id.view(batch_size, -1)
#
#
# embed_size = 300    # 词嵌入的维度
# hidden_size = 1024  # 使用RNN变种LSTM单元   LSTM的hidden size
# num_layers = 1      # 循环单元/LSTM单元的层数
# num_epochs = 5      # 迭代轮次
# num_samples = 1000  # 测试语言模型生成句子时的样本数
# batch_size = 20     # 一批样本的数量
# seq_length = 30     # 一个样本/序列长度
# learning_rate = 0.002   # 学习率
#
# corpus = Corpus()
# ids = corpus.get_data(glove_file, batch_size)
# vocab_size = len(corpus.dictionary)
# num_batches = ids.size(1) // seq_length
#
# # 有gpu的情况下使用gpu
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # RNN语言模型
# class RNNLM(nn.Module):  # RNNLM类继承nn.Module类
#     def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
#         super(RNNLM, self).__init__()
#         # 嵌入层 one-hot形式(vocab_size,1) -> (embed_size,1)
#         self.embed = nn.Embedding(vocab_size, embed_size)
#         # LSTM单元/循环单元
#         self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
#         # 输出层的全联接操作
#         self.linear = nn.Linear(hidden_size, vocab_size)
#
#     def forward(self, x, h):
#         # 词嵌入
#         x = self.embed(x)
#
#         # LSTM前向运算
#         out, (h, c) = self.lstm(x, h)
#
#         # 每个时间步骤上LSTM单元都会有一个输出，batch_size个样本并行计算(每个样本/序列长度一致)  out (batch_size,sequence_length,hidden_size)
#         # 把LSTM的输出结果变更为(batch_size*sequence_length, hidden_size)的维度
#         out = out.reshape(out.size(0) * out.size(1), out.size(2))
#         # 全连接
#         out = self.linear(out)  # (batch_size*sequence_length, hidden_size)->(batch_size*sequence_length, vacab_size)
#
#         return out, (h, c)
#
# model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
#
# # 损失构建与优化
# criterion = nn.CrossEntropyLoss() #交叉熵损失
# #使用Adam优化方法 最小化损失 优化更新模型参数
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#
# # 反向传播过程“截断”(不复制gradient)
# def detach(states):
#     return [state.detach() for state in states]
#
#
# # 训练模型
# for epoch in range(num_epochs):
#     # 初始化为0
#     states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
#               torch.zeros(num_layers, batch_size, hidden_size).to(device))
#
#     for i in range(0, ids.size(1) - seq_length, seq_length):
#         # 获取一个mini batch的输入和输出(标签)
#         inputs = ids[:, i:i + seq_length].to(device)
#         targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)  # 输出相对输入错一位，往后顺延一个单词
#
#         # 前向运算
#         states = detach(states)
#         outputs, states = model(inputs, states)
#         loss = criterion(outputs, targets.reshape(-1))
#
#         # 反向传播与优化
#         model.zero_grad()
#         loss.backward()
#         clip_grad_norm_(model.parameters(), 0.5)
#
#         step = (i + 1) // seq_length
#         if step % 100 == 0:
#             print('全量数据迭代轮次 [{}/{}], Step数[{}/{}], 损失Loss: {:.4f}, 困惑度/Perplexity: {:5.2f}'
#                   .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
#
# # 测试语言模型
# with torch.no_grad():
#     with open('sample.txt', 'w') as f:
#         # 初始化为0
#         state = (torch.zeros(num_layers, 1, hidden_size).to(device),
#                  torch.zeros(num_layers, 1, hidden_size).to(device))
#
#         # 随机选择一个词作为输入
#         prob = torch.ones(vocab_size)
#         input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)
#
#         for i in range(num_samples):
#             # 从输入词开始，基于语言模型前推计算
#             output, state = model(input, state)
#
#             # 做预测
#             prob = output.exp()
#             word_id = torch.multinomial(prob, num_samples=1).item()
#
#             # 填充预估结果（为下一次预估储备输入数据）
#             input.fill_(word_id)
#
#             # 写出输出结果
#             word = corpus.dictionary.idx2word[word_id]
#             word = '\n' if word == '<eos>' else word + ' '
#             f.write(word)
#
#             if (i + 1) % 100 == 0:
#                 print('生成了 [{}/{}] 个词，存储到 {}'.format(i + 1, num_samples, 'sample.txt'))
#
# # 存储模型的保存点(checkpoints)
# torch.save(model.state_dict(), 'model.ckpt')
#




















# word_list = ['man', 'piano']
# for word in word_list:
#     print(word)
#     for i in model.most_similar(word, topn=10):
#         print(i[0], i[1])


# def get_glove_info(glove_file_name):
#     with smart_open(glove_file_name) as f:
#         num_lines = sum(1 for _ in f)
#     with smart_open(glove_file_name) as f:
#         num_dims = len(f.readline().split()) - 1
#     return num_lines, num_dims
#
# def glove2word2vec(glove_input_file, word2vec_output_file):
#     num_lines, num_dims = get_glove_info(glove_input_file)
#     logger.info('converting %i vectors from %s to %s', num_lines, glove_input_file, word2vec_output_file)
#     with smart_open(word2vec_output_file, 'wb') as fout:
#         fout.write('{0}{1}\n'.format(num_lines, num_dims).encode('utf-8'))
#         with smart_open(glove_input_file, 'rb') as fin:
#             for line in fin:
#                 fout.write(line)
#     return num_lines, num_dims