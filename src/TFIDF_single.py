# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 15:49
# @Author  : Weiyang
# @File    : TFIDF_single.py

########################################################################################################################
# TF-IDF算法 抽取关键词
# 算法基本思想：求出每个候选关键词的TF-IDF值，然后按逆序输出；
# 算法输入：单词的IDF词典，文档
# 算法输出：
# word,TFIDF
# word,TFIDF
# ...

# 单文档的TFIDF关键词抽取算法
########################################################################################################################

import jieba as jb
import re

class TFIDF_single:
    '''单文档tfidf抽取关键词'''

    def __init__(self,word2idfpath='../idf/LCSTS2.0_word2idf'):
        self.word2idf = self.Readword2idf(word2idfpath) # word:idf

    def Readword2idf(self,word2idfpath):
        '''read word idf'''
        with open(word2idfpath,'r',encoding='utf-8') as fi:
            word2idf = {} # word:idf
            for line in fi:
                line = line.strip().split('\t')
                word2idf[line[0]] = word2idf.get(line[0],float("inf")) # 如果词典中没有单词的idf，则赋予无穷大
                word2idf[line[0]] = float(line[1])
            return word2idf

    def calculate_tf(self,filecontent):
        '''计算单词的词频tfidf , filecontent 文档内容'''
        word2tf = {} # word:tf 单词的词频
        word2tfidf = {} # word:tfidf 单词的tfidf
        line_lst = re.split("[。？！?!,，.;；]", filecontent)# 将文本切割成句子
        all_words = [] # 文档中的所有单词
        # 遍历每个句子
        for line in line_lst:
            words = jb.cut(line)
            # 记录单词的词频
            for word in words:
                # 去除非中文字符
                if '\u4e00' <= word <= '\u9fff':
                    word2tf[word] = word2tf.get(word,0)
                    word2tf[word] += 1
                    all_words.append(word)
        all_words = list(set(all_words))
        # 计算文档中每个单词的tfidf
        for word in all_words:
            word2tfidf[word] = word2tfidf.get(word,0)
            word2tfidf[word] = word2tf[word] * self.word2idf.get(word,float("inf"))
        return word2tfidf

    def extract_keywords(self,filecontent,n=5):
        '''抽取关键词'''
        word2tfidf = self.calculate_tf(filecontent)
        keywords = sorted(word2tfidf, key=word2tfidf.__getitem__, reverse=True)  # 按得分值排序

        for word in keywords[:n]:
            print('(',word,',', word2tfidf[word],')')

        print(keywords[:n])

if __name__=="__main__":
    t = TFIDF_single()
    filecontent = '今天有传在北京某小区，一光头明星因吸毒被捕的消息。' \
                  '下午北京警方官方微博发布声明通报情况，证实该明星为李代沫。' \
                  '李代沫伙同另外6人，于17日晚在北京朝阳区三里屯某小区的暂住地内吸食毒品，' \
                  '6人全部被警方抓获，且当事人对犯案实施供认不讳。'
    t.extract_keywords(filecontent,n=5)