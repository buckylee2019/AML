from __future__ import print_function
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
import tensorflow as tf
import pprint
import matplotlib.pyplot as plt
import math
import sys
import pandas as pd
import numpy as np
from numpy import exp, dot
import timeit
import matplotlib.pyplot as plt
import warnings
import time
from datetime import datetime
from collections import Counter
import re
import glob
import random
import sys
from gensim.models.word2vec import Word2Vec
model = Word2Vec.load('Parser/wordEmbedding/20180521_NPOS_SG_250_m20_w7.bin')
#data_utils.download_data_gdown("./") # gdrive-ckip
ws = WS("data")
pos = POS("data")
ner = NER("data")
keyword = pd.read_excel('data/犯罪關鍵字.xlsx')[['關鍵字']]

word_to_weight = {j:1 for j in keyword['關鍵字']}
   
newsdata = pd.DataFrame()
for filename in glob.glob('data/news/test_news.xlsx'):
    excel = pd.read_excel(filename)
    d = excel[['標題','內容','發佈時間','原始連結']]
    newsdata = pd.concat([newsdata,d])

newsdata.drop_duplicates(subset ="標題", keep = 'first', inplace = True) 
newsdata.drop_duplicates(subset ="內容", keep = 'first', inplace = True) 
newsdata.dropna(inplace = True)
dictionary = construct_dictionary(word_to_weight)

sentence_list = []
for s in newsdata['內容']:
    sentence_list.append(''.join(re.findall(r'[、。？！「，」:：；（）\/\\\u4e00-\u9fa5\w]+',str(s))))
word_sentence_list = ws(
    sentence_list,
    batch_sentences=2000,
    # sentence_segmentation=True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    recommend_dictionary = dictionary, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
)

df = newsdata
tmp_list = []
for i,news in enumerate(newsdata['標題']):
    
    
    words = word_sentence_list[i]
    length = len([ w for w in words if w in model])
    if length!=0:
        
        sim  = sum([model.similarity('犯罪',w) for w in words if w in model])/length
    
        if sim<=0.14:
            
            
            df = df[df['標題']!=news]
        else:
            tmp_list.append(words)
print('word seg done!')
pos_sentence_list = pos(tmp_list)
print('pos tag done!')
#entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
print('ner tag done!')

df['詞性'] = [' '.join(p) for p in pos_sentence_list]
df['斷詞'] =  [' '.join(w) for w in tmp_list]
# df['NER'] = ['' for _ in range(len(word_sentence_list))]
df.to_excel("data/2012-2019_news.xlsx")

