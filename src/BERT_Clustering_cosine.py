# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 20:20
# @Author  : Weiyang
# @File    : BERT_Clustering_distance.py

########################################################################################################################
# 基于BERT的词聚类关键词抽取算法
# 算法基本思想：
# 首先将文档进行分词，获取候选关键词，用于选取聚类中心词；
# 然后将文档中的词进行聚类并获取聚类中心词；
# 对文档进行词性标注，基于一定的模式(比如，0个或多个形容词跟随1个或者多个名词的词串作为名词短语)获取候选短语；
# 这里的模式是，我们抽取指定词性的单词，将其前后各一个词与其本身构成的子串作为候选短语；
# 最后，选择包含一个或多个聚类中心的短语作为最终的关键词。

# 聚类算法衡量相似性的标准是：余弦相似度
########################################################################################################################

from bert_serving.client import BertClient
import jieba as jb
from Kmeans_cosine import KMeans
import jieba.posseg as pseg
from collections import defaultdict
import re
import numpy as np

class BERT_Clustering:
    '''基于BERT的词聚类关键词抽取算法'''

    def __init__(self,n_cluster=10, patterns=['nr','ns','nt','nz','a','an','m','v'],stopwordspath='../stopwords/百度停用词表.txt'):
        self.stopwords = self.Readstopwords(stopwordspath)  # 停用词
        self.n_cluster = n_cluster # 聚类的簇数
        self.patterns = patterns # [pattern1,pattern2,...] 用于抽取候选短语的词性搭配模式，比如[名词,形容词,..]
        # patterns 待抽取的词性，最后输出的是：以该词性的词为中心词，前后各取一个词构成子串

    def Readstopwords(self, stopwordspath):
        '''read stopwords'''
        with open(stopwordspath, 'r', encoding='utf-8-sig') as fi:
            stopwords = []
            for line in fi:
                line = line.strip()
                stopwords.append(line)
            return stopwords

    def Readcontent(self,filecontent):
        '''将文本切割成句子'''
        line_lst = re.split("[。？！?!,，.;；]",filecontent)
        all_words = [] # 候选词，用于聚类，选取聚类中心词
        for line in line_lst:
            # 分词
            words = jb.cut(line)
            # 词性标注
            #word_flag = pseg.cut(line)
            # 去除非中文字符
            words = [word for word in words if '\u4e00' <= word <= '\u9fff']
            all_words.extend(words)
        # 过滤重复词
        all_words = list(set(all_words))
        return line_lst,all_words # 句子序列，单词序列

    def getPointsCosine(self,points1,points2):
        '''获取点集points1与points2中每两个点之间的余弦相似度,原理:Cos(A,B) = A*B/|A||B|'''
        A, B = np.array(points1[:]), np.array(points2[:])
        BT = np.transpose(B)
        A_BT = np.dot(A, BT)
        Asq = A ** 2
        Asq = np.tile(np.sum(Asq, axis=1, keepdims=True), (1, A_BT.shape[1]))
        Bsq = BT ** 2
        Bsq = np.tile(np.sum(Bsq, axis=0, keepdims=True), (A_BT.shape[0], 1))
        Cosine = A_BT / (np.sqrt(Asq) * np.sqrt(Bsq)) # 两个点集中每两个点的余弦相似度
        return Cosine

    def calculate_similarity(self,all_words):
        '''
        对文档中的单词进行聚类，并获取聚类中心词
        all_words：[word1,word2,...]
        '''
        bc = BertClient()
        # 存储每个单词的词向量
        points = bc.encode(all_words).tolist()
        # 开始聚类
        k = KMeans(n_clusters=self.n_cluster)
        # 每个单词的类别
        labels = k.fit_predict(points)
        # 聚类中心
        centers = k.cluster
        # 获取每个聚类中心与每个单词的余弦相似度
        distance = self.getPointsCosine(centers,points)
        # 获取与每个聚类中心余弦相似度最大的单词，即与聚类中心最相似的单词
        indexs = [np.argmax(distance[center]) for center in range(self.n_cluster)]
        # 与每个聚类中心最相似的单词
        center_words = [all_words[index] for index in indexs]
        # 获取每个类中的单词
        cluster_words = defaultdict(list)
        for index,label in enumerate(labels):
            cluster_words[label].append(all_words[index])
        return center_words,labels ,cluster_words # 聚类中心的单词，每个单词的类别,每个类中的单词

    def getPosseg(self,lines):
        '''
        基于一定模式抽取词性搭配的短语作为候选短语，lines是句子列表
        '''
        # 存储符合条件的候选短语
        phrases = []
        # 遍历句子
        for line in lines:
            # 对句子进行词性标注
            result = pseg.cut(line)
            # 单词序列
            words = []
            # 词性序列
            psgs = []
            # 获取词性标注的结果
            for w in result:
                words.append(w.word)
                psgs.append(w.flag)
            # 遍历单词序列和词性序列
            previous = '' # 存储当前词的前一个单词
            count = 0
            for word,flag in zip(words,psgs):
                # 如果当前词的词性在patterns中，则抽取当前词前后各一个词构成子串
                if flag in self.patterns:
                    # 如果当前词不是末尾词
                    if count < len(words)-1:
                        subString = previous+word+words[count+1]
                    else:
                        subString = previous+word
                    # 将符合条件的子串加入到候选短语中
                    phrases.append(subString)
                # 移动前一个单词
                previous = word
                count += 1
        # 过滤重复词
        phrases = list(set(phrases))
        return phrases # 返回候选短语

    def extract_Keywords(self,filecontent):
        '''抽取关键词,filecontent 待抽取的文档内容'''
        # 文档的句子列表，不重复的单词列表
        lines,words = self.Readcontent(filecontent)
        # 读取停用词
        stopwords = self.Readstopwords()
        # 去除停用词
        words = [word for word in words if word not in stopwords]
        # 获取中心词
        center_words, labels, clusters = bt.calculate_similarity(words)
        # 获取候选短语列表
        phrases = self.getPosseg(lines)
        keywords = [] # 存储关键词
        # 遍历候选短语
        for phrase in phrases:
            # 遍历中心词，查看候选短语中是否包含中心词，如果包含，则输出
            for center in center_words:
                if center in phrase:
                    keywords.append(phrase)
                    break
        return keywords

if __name__ == '__main__':
    filecontent = '韩媒称，国际油价暴跌让中国笑逐颜开。' \
                  '分析称，中国在国际油价暴跌后“三管齐下”，大幅缩减石油进口费用、扩大战略储备油，' \
                  '同时向资金短缺的产油国提供贷款以壮大亲中势力。美银美林指出，国际油价每下跌10%，中国GDP会增长0.15%。'
    bt = BERT_Clustering()
    keywords = bt.extract_Keywords(filecontent)
    print(keywords)