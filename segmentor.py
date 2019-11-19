from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from Parser.preprocess import *
import re
ws = WS('data/')
pos = POS('data/')
def segment(sentence):
    string = ''.join(re.findall(r'[\u4e00-\u9fa5：\d\.\，\。\ \？\！\、\；\,\;]',str(sentence)))
    
    sent = re.split('，|。|\ |？|！|、|；|,|;|：',string)
    word_sentence_list = ws(
        sent,
        batch_sentences=2000,
        # sentence_segmentation=True, # To consider delimiters
        # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
        #recommend_dictionary = dictionary, # words in this dictionary are encouraged
        # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )

    pos_sentence_list = pos(word_sentence_list)
    allcont = []
    for W,P in zip(word_sentence_list,pos_sentence_list):
        
        decodeOut = ' '.join([s+'_'+p for s,p in zip(W,P)])
        decodeOut = re.sub(r'\_\w*CATEGORY','_PM',decodeOut)
        
        allcont.append(decodeOut)
    return ';'.join(allcont)

if __name__ == '__main__':
    while(1):
        i = input()
        print(segment(i))
