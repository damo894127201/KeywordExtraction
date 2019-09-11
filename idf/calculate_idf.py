# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 14:09
# @Author  : Weiyang
# @File    : calculate_idf.py

########################################################################################################################
# 计算语料库中每个单词的idf值，并输出
# 语料库的格式：每行一篇文档
########################################################################################################################

import math
import jieba as jb

document_count = 0 # 语料中的文档总数
words = set() # 语料中的单词类别个数
word2count = {} # word:count 包含某个单词的文档数量
word2idf = {} # word:idf 单词的idf值

########################################## 读取数据 ##########################################

# 语料库路径
corpuspath = 'Article_III.txt'
fi = open(corpuspath,'r',encoding='utf-8-sig')
print('Reading corpushpath .....')
# 遍历语料
for article in fi:
    # 判断文档是否为空
    if len(article.strip()) == 0:
        continue
    document_count += 1
    word_lst = jb.cut(article)
    temp = [] # 存储当前文章内的中文单词
    for word in word_lst:
        # 去除非中文字符
        if '\u4e00' <= word <= '\u9fff':
            words.add(word)
            temp.append(word)
    # 过滤重复单词
    word_lst = set(temp)
    # 将文章内的单词对应的word2count 加1
    for word in word_lst:
        word2count[word] = word2count.get(word,0)
        word2count[word] += 1
fi.close()

######################################### 计算idf #############################################
# 输出单词的idf值
fi = open('./word2idf','w',encoding='utf-8-sig')
for word in words:
    word2idf[word] = math.log(float(document_count)/(word2count[word]+1))
    fi.write(word+'\t'+str(word2idf[word])+'\n')
fi.close()