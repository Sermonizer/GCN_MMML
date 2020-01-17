import json
import sys
import pandas as pd
import numpy as np
import gensim

sys.path.append('/home/administer/Tengxu/Code/Text_produce/SceneGraphParser')
import sng_parser
from pprint import pprint

pd.set_option('display.max_rows', None)
# flickr30k = json.load(open('/media/administer/新加卷/DataSet/Multi modal/Flickr30k/flickr30k/dataset.json', 'r'))
# sentense = flickr30k['sentences']

# dicks
dicks = []
lenth = []

def embed_glove(sentences):
    with open('/home/administer/Tengxu/DataSet/Glove/glove.6B/glove.6B.300d.txt') as file:
        embedding = []
        vocab = []
        word_index = []
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embedding.append(row[1:])
    embedding = np.asarray(embedding, dtype='float32')
    for word in sentences:
        for word1 in vocab:
            if word == word1:
                word_index.append(vocab.index(word1))
        print(word_index)
    return word_index

# make parser
with open('/media/administer/新加卷/DataSet/Multi modal/Flickr30k/flickr30k/dataset.json') as f:
    line = f.readline()
    flickr = json.loads(line)
    num = 0
    for i in flickr['images']:
        # dicks = []
        # print(i)
        for j in range(5):
            raw = i['sentences'][j]['raw']
            # print(raw)
            # graph = sng_parser.parse(raw)
            # dicks.append([])

            embed_glove(i['sentences'][j]['tokens'])

            # for k in graph['entities']:
            #     dicks[num].append(str.lower(k['span']))
            #     # dicks.append(k['span'])
            # num += 1

    # print(dicks)
        # dicks.append([])
        # add = {'attributes:': dicks}
        # # json.dump(add, f)
        # tokens = i['sentences'][0]['tokens']
        # lenth.append(len(tokens))

# lenth.sort(reverse=True)
# print(lenth[:40])



# if __name__ == '__main__':




