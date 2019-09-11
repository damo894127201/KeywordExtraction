# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 15:49
# @Author  : Weiyang
# @File    : TFIDF_multi.py

########################################################################################################################
# TF-IDF算法 抽取关键词
# 算法基本思想：求出每个候选关键词的TF-IDF值，然后按逆序输出；
# 算法输入：语料库中的文档，格式为：每行一篇文档
# 算法输出：
# word,TFIDF
# word,TFIDF
# ...

# 多文档的TFIDF关键词抽取算法
########################################################################################################################

import os
import math
import jieba
import nltk
import re


class TFIDF_multi:
    def __init__(self):
        self.corpus_path='' #用于保存语料库的路径
        self.name=[] #保存语料库里每个文档的文件名
        self.doc={} #用来保存文章内容,key为文章名称，value为文章内容
        self.count={} #计数每个单词的频数
        self.tf={} #存放每个文档中每个单词的tf-idf值
        self.tf_idf={} #计数每个文档中每个单词的tf-idf值
        self.stopwords=[] #用于加载外部停用词词典
        self.doc_count=0 #语料库里总文档数量

    #读取语料库里的所有文件名,同时读取停用词
    def ReadCorpus(self,corpus_path,stopwords_path):
        #获取语料库里所有文件名
        self.name=os.listdir(corpus_path)
        self.doc_count=float(len(self.name)) #这里将数转为浮点数，便于后面计算tf-idf的值，因为整数除整数，会切掉小数部分，避免值为0
        self.corpus_path=corpus_path
        #读取停用词
        with open(stopwords_path,'r') as f :
            for line in f.readlines():
                self.stopwords.append(line)
        #根据运行结果，将停用词典里不存在的不重要的词添加进去
        stoplist=['出来','','0','应当','做出','?','-','而言','那么','姓','就要','一','由于','〇','○','—','力','']
        self.stopwords=self.stopwords+stoplist

    #读取文档，分词后，同时去除停用词
    def ReadFile(self):
        for name in self.name:
            #打开路径为corpus的路径+文件名
            with open(self.corpus_path+'/'+name,'r') as f:
                #空字符串，用于保存文章内容
                content_str=''
                for line in f.readlines():
                    line1=line.strip() #去除首尾空格
                    re_pattern=re.compile(r'(\D\S)*') #去除空格和数字
                    line2=re_pattern.match(line1)
                    if line2==None:
                        continue
                    content_str+=line2.group() #获取正则表达式匹配后的结果的字符串
                #对文章内容进行分词
                split_word=jieba.cut_for_search(content_str)#搜索引擎模式分词
                #split_word=jieba.cut(content_str,cut_all=False) #精确模式分词
                #split_word=jieba.cut(content_str,cut_all=True)#全模式分词
                #split_word=jieba.cut(content_str,cut_all=False,HMM=True) #分词时使用HMM模型

                #剔除停用词
                #一个空列表，用于存放剔除停用词后的关键词
                keyword=[]
                for word in split_word:
                    if word not in self.stopwords:
                        keyword.append(word)
                #将文档内容添加到self.doc中
                self.doc[name]=keyword

    #计算每个文档中每个词的tf-idf值
    def calculate_tf_idf(self):
        #计算每个文档中每个单词的tf值
        #获取每个文档的名字和其内容
        for name,doc in self.doc.items():
            #获取文档内容doc里每个关键词的频数，这里使用nltk.FreqDist方法,
            # 返回的是一个词典，key是词，value是频数
            word_freq=nltk.FreqDist(doc)
            #文档总词数
            word_count=len(doc)
            #存放每个文档的tf值,key是单词,value是词的频率
            tf={}
            #计算tf值
            for word,freq in word_freq.items():
                #单词的tf值
                tf[word]=freq/word_count
            self.tf[name]=tf #此处可知self.tf 的 key 是文档名,value是一个词典，词典里存储的是单词：tf值

        #计算每个文档中每个单词的tf-idf值
        #首先要计算n包含每个单词的文档总数
        for name,doc in self.doc.items():
            #用于存放某个文档的tf-idf值,key是Word,value是tf-idf
            tf_idf={}

            #获取文档中的单词
            for word in doc:
                # 用于计数包含每个单词的文档总数
                word_count_doc = 0
                for name1,doc1 in self.doc.items():
                    if word in doc1:
                        word_count_doc+=1
                #计算文档中单词的idf值
                idf=math.log(self.doc_count/(word_count_doc+1))
                #计算文档中单词的tf-idf值
                tf_idf[word]=self.tf[name][word]*idf
            #存放文档的tf_idf值
            self.tf_idf[name]=tf_idf

    #筛选出每个文档的前n个最大的tf-idf的关键词，并输出
    def Select_n_keywords(self,n):
        #遍历self.tf_idf，获取每个文档的前n个最大的tf-idf最大的关键词
        for name ,tf_idf in self.tf_idf.items():
            keywords=sorted(tf_idf,key=tf_idf.__getitem__,reverse=True)
            print(name+" 的前%d个重要的关键词是:"%n)
            print(keywords[:n])
            print('\n')


if __name__=="__main__":
    extractkeywords=TFIDF_multi()
    extractkeywords.ReadCorpus('/home/weiyang/MyRepository/corpus','/home/weiyang/MyRepository/stopwords/stopword.txt')
    extractkeywords.ReadFile()
    extractkeywords.calculate_tf_idf()
    extractkeywords.Select_n_keywords(20)