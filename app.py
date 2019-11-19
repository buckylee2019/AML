import pandas as pd
import glob
from collections import defaultdict
from dateutil import parser as tparser
import json
import re
import copy
from fuzzywuzzy import fuzz
from datetime import datetime
from gensim.models.word2vec import Word2Vec
import flask
import pycnnum as cnnum
import Parser.dependencyParser as parser
from flask import jsonify, request
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.corpora import Dictionary
from gensim.similarities import MatrixSimilarity, WmdSimilarity, SoftCosineSimilarity
from gensim.models import TfidfModel
import numpy as np
app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False


model = Word2Vec.load('Parser/wordEmbedding/20180521_NPOS_SG_250_m20_w7.bin')
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

ws = WS("Parser/dataParser")
pos = POS("Parser/dataParser")
ner = NER("Parser/dataParser")

gender = {}
identifier = defaultdict(dict)

with open('Parser/dataParser/genderName.txt') as f:
    for l in f.readlines():
        name,sex = l.split(',')[0],l.split(',')[1].strip('\n')
        gender[name] = sex
class intergrateCluster():
    def  __init__(self,group,cluster):
        self.group = group
        self.cluster = cluster
        self.newsid = [news['id'] for news in self.group if news['id'] not in self.cluster['None']]
        self.documents = [news['info_1']['crime_keys'] for news in self.group if news['id'] in self.newsid]


        self.dictionary = Dictionary(self.documents)
        self.tfidf = TfidfModel(dictionary=self.dictionary)

        similarity_index = WordEmbeddingSimilarityIndex(model.wv)
        self.similarity_matrix = SparseTermSimilarityMatrix(similarity_index, self.dictionary)
    
        
    def softcossim(self,query, documents):
        # Compute Soft Cosine Measure between the query and the documents.
        query = self.tfidf[self.dictionary.doc2bow(query)]
        index = SoftCosineSimilarity(
            self.tfidf[[self.dictionary.doc2bow(document) for document in self.documents]],
            self.similarity_matrix)
        similarities = index[query]
        return similarities
    def run_combine(self):
        for n in self.group:
            if n['id'] not in self.newsid:


                

                ans = self.softcossim(n['info_1']['crime_keys'],self.documents)
               
                for k in self.cluster:
                    if self.newsid[np.argsort(ans)[-1]] in self.cluster[k]:
                        self.cluster[k].append(n['id'])
        del self.cluster['None']



def getPersonInfo(NAME,dp,date):
    #person = newsdata[newsdata['內容'].str.contains(NAME, na=False)]
    
    res = defaultdict(list)

    
    for data in dp:


        stack = []
        for j in range(len(data['words'])):
            if data['words'][j]['text'] == NAME:

                stack = [(NAME,j)]
        
        if not stack:
            continue
        seen = set()
        seen.add(stack[0])
 
        while(stack):
            
                
            tmp = stack.pop()
            
            for rel in data['arcs']:
            
             
                if rel['dir'] == 'left':
                    
                    if data['words'][rel['end']]['text']== tmp[0] and rel['start'] != tmp[1]:
                        
                        if (data['words'][rel['start']]['text'],rel['start']) not in seen:
                            
                            stack.append((data['words'][rel['start']]['text'],rel['start']))
                            seen.add((data['words'][rel['start']]['text'],rel['start']))
                            res['modifier'].append((data['words'][rel['start']]['text'],data['words'][rel['start']]['tag'],rel['start'])) 
                else:
                    if data['words'][rel['start']]['text']== tmp[0] and rel['end']!=tmp[1]:
                        if (data['words'][rel['end']]['text'],rel['end']) not in seen:
                            seen.add((data['words'][rel['end']]['text'],rel['end']))
                            stack.append((data['words'][rel['end']]['text'],rel['end']))
                            res['modifier'].append((data['words'][rel['end']]['text'],data['words'][rel['end']]['tag'],rel['end']))
        
    if res:
        res['modifier'] = list(set(res['modifier']))
        if type(date)==str:
            res['time'] = tparser.parse(date).year
        else:
            res['time'] = date.year
        
    return res


def find_crime(words):
    crime_key = ['貪污','洗錢','人身攻擊','詐欺','賄賂','犯罪','詐騙','恐嚇']
    list_c = []
    for w in words:
        if model.wv.__contains__(w) and len(w)>=2:
            for c in crime_key:
                if model.wv.similarity(w,c)>0.5:
                    list_c.append(w)
                    break
    return list(set(list_c))



def identify(name,whole,org):
    titleNew = {}
    
    titleNew['name'] = name
    industry = {'金融':0,'科技':0,'製造業':0,'娛樂業':0,'餐飲':0,'旅遊':0,'紡織業':0,'電子':0,'失業':0,'軍警':0,'醫療':0,'政治':0,'教育':0,'法律':0,'新聞':0}
    titleNew['industry'] = {}
    hit = 0
    for key in whole['modifier']:
        if key[1] == 'Nb' or key[1] =='Na' :
            if model.wv.__contains__(key[0]):
                hit += 1
                industry = {c:(model.wv.similarity(key[0],c)+industry[c]) for c in industry}
        
        for o in org:
            if key[1] == 'Nc' and key[0] in o:
             
                titleNew['location'] = o
                break

        if key[0] in gender:
            #print(personB)
            titleNew['gender'] = gender[key[0]]
        
        if '歲' in key[0]:
            age = key[0].replace('歲','')
            age = age.replace('廿','二十')
            age = age.replace('卅','三十')
            age = age.replace('卌”','四十')
            year = whole['time']
            if re.search(r'\d',age):
                titleNew['birth_date'] = year - int(re.findall(r'\d+',age)[0])

            else:

                titleNew['birth_date'] = year - cnnum.cn2num(age)
                
    if 'gender' not in titleNew:
        titleNew['gender'] = 'None'
    if 'birth_date' not in titleNew:
        titleNew['birth_date'] = 'None'
    if 'location' not in titleNew:
        titleNew['location'] = 'None'
    if hit!=0:
        industry = {c:industry[c]/hit for c in industry}
        titleNew['industry'] = sorted(industry.items(), key=lambda kv: kv[1],reverse = True)[0]
    elif hit==0:
        titleNew['industry'] = ('None',0.0)


    
    return titleNew


wholeP = defaultdict(dict)


# for i,NAME in enumerate(nameList):

#     print(NAME)
#     if NAME not in wholeP:
@app.route('/textcloud', methods=['POST'])
def clustering():
    data = request.get_json()
    target = data['searchKeys']['name']
    group = data['hit_news']
    df = pd.DataFrame(group)
    df['post_time']=pd.to_datetime(df['post_time'])
    df = df.set_index('post_time')
    idgroupbytime = df.resample('MS').apply(lambda x: x['id'])
    time_cluster = defaultdict(list)
    for k in idgroupbytime.keys():

        if type(idgroupbytime[k])!=str:
            for i in idgroupbytime[k[0]]:
                time_cluster[k[0]].append(i)
        else:
            time_cluster[k[0]].append(idgroupbytime[k])

        time_cluster[k[0]] = list(set(time_cluster[k[0]]))
        
        
    gender_cluster = defaultdict(list)
    location_cluster = defaultdict(list)
    industry_cluster = defaultdict(list)
    age_cluster = defaultdict(list)
    crime_cluster = defaultdict(list)
    seen_key = {}
    for info in group:
        

        for p in info['info_1']['involved_parties']:
            if p['name']==target and len(info['info_1']['crime_keys'])>=2 :
                if tuple(info['info_1']['crime_keys']) in seen_key:
                    seen_p = seen_key[tuple(info['info_1']['crime_keys'])]
                    gender_cluster[seen_p['gender']].append(info['id'])
                    location_cluster[seen_p['location']].append(info['id'])
                    industry_cluster[seen_p['industry'][0]].append(info['id'])
                    age_cluster[seen_p['birth_date']].append(info['id'])
                else:
                    seen_key[tuple(info['info_1']['crime_keys'])] = p
                    gender_cluster[p['gender']].append(info['id'])
                    location_cluster[p['location']].append(info['id'])
                    industry_cluster[p['industry'][0]].append(info['id'])
                    age_cluster[p['birth_date']].append(info['id'])
    
    location = intergrateCluster(group,location_cluster)
    location.run_combine()
    news_group=defaultdict(list)

    locat = copy.deepcopy(location.cluster)
    loc_seen = []
    new_loc_cluster = defaultdict(list)
    for k in locat.keys():
        if k not in loc_seen:
            new_loc_cluster[k] = locat[k]

        for p in locat.keys():
            if fuzz.ratio(k,p)>70 and fuzz.ratio(k,p)!=100:
                if p not in loc_seen:

                    loc_seen.append(p)
                    loc_seen.append(k)
                    new_loc_cluster[k]+=locat[p]
    
    
    
    for i,loc in enumerate(new_loc_cluster):
        news_hit = {}
        news_hit['id'] = 'cluster_'+str(i)
        news_hit['text_cloud'] = []
        news_hit['crime_type'] = ''
        news_hit['news_sources'] = []
        locgroup = [n for n in group if n['id'] in new_loc_cluster[loc]]
        news_hit['hit_news']=locgroup
        news_group['groups'].append(news_hit)

    return jsonify(news_group)
@app.route('/newsAnalysis', methods=['POST'])
def wordparsing():
    data = request.get_json()
    crime_news = defaultdict(list)
    for news in data['news']:
        print(news['title'])
        

        sentence_list=[''.join(re.findall(r'[、。？！「，」:：；（）\/\\\u4e00-\u9fa5\w]+',str(news['content'])))]
        word_sentence_list = ws(sentence_list)
        words = word_sentence_list[0]
        length = len([ w for w in words if model.wv.__contains__(w)])
        if length!=0:
        
            sim  = sum([model.wv.similarity('犯罪',w) for w in words if model.wv.__contains__(w)])/length
    
        if sim>0.14:
            
            news_info = news
            pos_list = pos(word_sentence_list)
            ner_list = ner(word_sentence_list,pos_list)
            nstring = ' '.join([s+'_'+p for s,p in zip(word_sentence_list[0],pos_list[0])])
            nstring = re.sub(r'\_\w*CATEGORY','_PM',nstring)
      
        
            sent = re.split(r'\ [\ |，|：|。|；|！|？]_PM ',nstring)
            
            dp = parser.quickparse(sent)
            
            person = [person[3] for person in ner_list[0] if 'PERSON' in person]
            organization = [org[3] for org in ner_list[0] if 'ORG' in org]
            person = list(set([p for p in person if len(p)>=2]))
            
            parties =[]
            for involve in person:
                P = getPersonInfo(involve,dp,news['post_time'])              
                parties.append(identify(involve,P,organization))
            crime_key = find_crime(word_sentence_list[0])
            news_info['info_1'] = {}
            
            news_info['info_1']['involved_parties'] = parties
            #news_info['info_1']['segment_words'] = words
            news_info['info_1']['segment_title'] = ws([news['title']])[0]
            news_info['info_1']['crime_keys'] = crime_key
            news_info['info_1']['post_time'] = news['post_time']
            crime_news['news'].append(news_info)
            


#     person = newsdata[newsdata['內容'].str.contains(name, na=False)]
#     wholeP = getPersonInfo(name,person)
    #print(request.get_json())
    

    return jsonify(crime_news)

app.run()
