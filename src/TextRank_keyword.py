# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 15:49
# @Author  : Weiyang
# @File    : TextRank_keyword.py

########################################################################################################################
# textRank算法抽取关键词
# pageRank算法基本思想：重要的网页被重要的网页连接；网页的pageRank值本质上是用户访问这个网页的概率；
# textRank算法使用PageRank计算PR值的公式，构造了一个词与词之间的有向图，所有在窗口内的单词都是窗口第一个单词的出链节点；
# 算法输入：文档内容，由句子序列构成，句子之间以 \t 分割，每个句子格式为：word1 word2 word3....
# 算法输出：关键词序列，格式为：word1 word2 word3 ....
########################################################################################################################
import numpy as np
import re

class TextRank_Keyword(object):
    '创建TextRank类用于抽取关键词'

    def __init__(self,sentences,N,window_size=5,step_size=1):
        self.sentence = re.sub('\t',' ',sentences) #存储输入的文本内容,将其中的\t符号置换为空格
        self.window_size = window_size # 移动窗口的大小，默认值为5
        self.step_size = step_size # 窗口移动的步长，默认为1
        self.N = N # N为输出多少个关键词
        self.word_index = {} # 给文本内容中的单词以数字索引，格式为：{word1:index1,word2:index2,...}

    '创建单词与单词之间的连接词典links，格式为：{word1:set([word2,word3,..]),word2:set([word3,word2,...]),...}'
    '单词word1对应的value值表示为，从单词word1可连接到的所有单词'
    def build_links(self):
        word_lst = self.sentence.split()
        links = {word:set() for word in word_lst}
        for i in range(0,len(word_lst),self.step_size):
            links[word_lst[i]].update(word_lst[i+1:self.window_size+i])
        return links

    '给单词增加索引，格式为:word:index ,用于构建词与词之间的邻接矩阵adjaceny matrix'
    def build_index(self):
        word_lst = set(self.sentence.split())
        return {word:index for index,word in enumerate(word_lst)}

    '构建词与词之间的邻接矩阵，即转移概率矩阵，其中matrix[i][j]表示从单词i到单词j的转移概率'
    def build_transition_matrix(self):
        links = self.build_links() # 单词与单词之间的连接词典
        self.word_index = self.build_index() # 单词到数字索引，格式为：word:index
        word_lst = set(self.sentence.split()) # 单词的种类个数
        matrix = np.zeros((len(word_lst),len(word_lst))) # 创建转移概率矩阵，初始值为0
        for word in links:
            #如果从当前单词无法转移到其它单词，即word:set()，这样的单词称为悬空单词dangling word
            if not links[word]: # not set() 为True
                # 设置当前单词转移到其它所有单词的概率相等，即1/N ，这里N是文档中的所有单词总数
                matrix[self.word_index[word]] = np.ones(len(word_lst))/float(len(word_lst))
            else:
                for sub_word in links[word]:
                    # 设置当前单词转移到它所连接到的单词的概率相等，即1/N，这里的N是当前单词连接到的单词总数
                    matrix[self.word_index[word]][self.word_index[sub_word]] = 1.0/len(links[word])
        return matrix

    'pageRank算法实现，有两个重要参数：eps,表示若相邻两次迭代之间各个PR值的差值之和小于eps，则停止迭代；'
    'd:阻尼系数，经验值设置为0.85，它在PageRank中表示用户将会以1-d的概率随机选择某个网页作为下一个网页，这并不考虑网页之间的连接关系'
    def pageRank(self,eps=0.0001,d=0.85):
        matrix = self.build_transition_matrix()
        PR = np.ones(len(matrix))/len(matrix) # 每个单词初始PR值设置为相等，即1/N，N为文档中所有单词的总数
        #循环迭代
        while True:
            # 更新所有单词的PR值
            # matrix.T 每一行表示所有对单词i有出链的单词到i的概率，且概率相等，为1/N,N为单词i的出度，即单词i连接到的单词总数
            new_PR = np.ones(len(matrix)) * (1-d)/len(matrix) + d * matrix.T.dot(PR)
            delta = abs(new_PR - PR).sum() # 相邻两次迭代之间的PR差值之和
            if delta <= eps:
                return new_PR
            PR = new_PR

    '输出N个关键词，关键词以PR值倒排'
    def getKeyword(self):
        PR = self.pageRank()
        keyword_index = [item[0] for item in sorted(enumerate(PR),key=lambda item:-item[1])]
        index_word = {self.word_index[word]:word for word in self.word_index} # 数字到单词，格式为：index:word
        keyword = [index_word[index] for index in keyword_index]
        return keyword[:self.N]
if __name__ == '__main__':
    line = '逐一 甄别 滞留 站 送 异地 安置 家属 抚慰金 每名 遇难者 低于 元 条例 灾害 救助 提供 救灾 生活 物资 生活费 补助'
    t = TextRank_Keyword(line,5)
    print(t.getKeyword())