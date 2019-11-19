
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
import Parser.dependencyParser as parser
import glob
import random
import re
from collections import defaultdict
from dateutil import parser as tparser

print(type(sys.argv[1]))
newsdata = pd.DataFrame()
for filename in glob.glob("data/2012-2019_news.xlsx"):
    excel = pd.read_excel(filename)
    d = excel[['標題','內容','發佈時間','斷詞','詞性']]
    newsdata = pd.concat([newsdata,d])

newsdata.drop_duplicates(subset ="標題", keep = 'first', inplace = True) 
newsdata.drop_duplicates(subset ="內容", keep = 'first', inplace = True) 
newsdata.dropna(inplace =True)


if sys.argv[1] == '1':
    
    rang = range(50000)
    df = newsdata[['標題','內容','發佈時間']][:50000]
    
elif sys.argv[1] == '2':
    df = newsdata[['標題','內容','發佈時間']][50000:100000]
    rang = range(50000,100000)
else:
    df = newsdata[['標題','內容','發佈時間']][132350:]
    rang = range(132350,len(newsdata['斷詞'].tolist()))
seg = newsdata['斷詞'].tolist()
postag = newsdata['詞性'].tolist()
dpout = []
for i in rang:
    print('Article number: ',i)
    if str(seg[i]) != 'nan':
        
        nstring = ' '.join([s+'_'+p for s,p in zip(seg[i].split(),postag[i].split())])
        nstring = re.sub(r'\_\w*CATEGORY','_PM',nstring)
      
        
        sent = re.split(r'\ [\ |，|：|。|；|！|？]_PM ',nstring)

        dp = parser.quickparse(sent)

    dpout.append(dp)


df['剖析樹']=dpout

df.to_json('data/news_parsed'+sys.argv[1]+'.json')
