# -*- coding: utf-8 -*-
# @Time    : 2017/5/4 15:49
# @Author  : Weiyang
# @File    : Rake_raw.py

########################################################################################################################
# Rake 算法抽取关键词(短语)
# Rake(快速自动关键词抽取)算法基本思想：对候选关键词打分，然后排序输出得分最高的关键词；候选关键词的得分由组成它的每个字
# 的得分累加而得，每个字的得分由 该字的度/该字的词频 得到；每个字的度，是指该字与文档中所有字在候选关键词中的共现次数，
# 具体计算如下，该字每与一个字共现在一个候选关键词中，度加1，考虑该字本身；因此，当与该字共现的字越多，该字的度就越大；
# 算法输入：原始文档内容
# 算法输出：关键词序列，格式为：
# word1 ,score
# word2 ,score
# word3 ,score
# ....

# Rake_raw.py 遵循论文中的思想，以停用词来切割句子生成候选关键词；
########################################################################################################################

import jieba as jb
import re

class Rake:
    '快速自动关键词抽取算法'
    def __init__(self,stopwordspath='../stopwords/百度停用词表.txt'):
        self.stopwords = self.Readstopwords(stopwordspath) # 停用词

    def Readstopwords(self,stopwordspath):
        '''read stopwords'''
        with open(stopwordspath,'r',encoding='utf-8-sig') as fi:
            stopwords = []
            for line in fi:
                line = line.strip()
                stopwords.append(line)
            return stopwords

    def Readcontent(self,filecontent):
        '''将文本切割成句子'''
        line_lst = re.split("[。？！?!,，.;；]",filecontent)
        content = [] # 存储文档句子

        '''将每个句子以停用词作为分隔符，切割成多个短语,构建候选词集'''
        for line in line_lst:
            '''将句子切词'''
            line = jb.cut(line)
            sline = ''
            for word in line:
                # 去除数字
                if word.isdigit():
                    continue
                if word in self.stopwords:
                    word = "#" # 将停用词作特殊标记
                    sline += word
                else:
                    sline += word
            line = sline.split("#")
            content.append(line)
        return content #[[a,b,..],[s,d,..],..] 分句后的文档内容
    
    def calculate_wordScore(self,content):
        '''计算每个字(word)的得分'''
        '''content 分句后的文档，格式为[[a,b,..],[s,d,..],..]'''
        word_Frequency = {} #每个字的词频
        word_Degree = {} #每个字的度
        wordScore = {} # 每个字的得分

        for line in content:
            for phrase in line:
                # 候选关键词包含的字数
                phrase_length = len(phrase) - 1  # 短语的长度-1
                for word in phrase:
                    # 计算每个字的词频
                    word_Frequency[word] = word_Frequency.get(word,0)
                    word_Frequency[word] += 1
                    #计算每个字的度
                    word_Degree[word] = word_Degree.get(word, 0)
                    word_Degree[word] += phrase_length
         #考虑该字本身
        for word in word_Frequency:
            word_Degree[word] = word_Degree[word] + word_Frequency[word]
        #计算每个字的得分值
        for word in word_Frequency:
            word_Score = float(word_Degree[word])/word_Frequency[word]
            wordScore[word] = word_Score
        return wordScore # 每个字的得分

    def calculate_phraseScore(self,content,wordScore):
        '''计算短语的得分值'''
        '''content 分句后的文档，格式为[[a,b,..],[s,d,..],..]; wordScore 每个字的得分词典'''
        phraseScore = {} # 候选关键词的得分
        for line in content:
            for phrase in line:
                for word in phrase:
                    phraseScore[phrase] = phraseScore.get(phrase,0)
                    phraseScore[phrase] += wordScore[word]
        return phraseScore

    def extract_keywords(self,filecontent,n=5):
        '''抽取前n个得分值最大的关键词短语'''
        content = self.Readcontent(filecontent) #[[a,b,..],[s,d,..],..] 分句后的文档内容
        wordScore = self.calculate_wordScore(content) # 每个字的得分
        phraseScore = self.calculate_phraseScore(content,wordScore) # 每个候选关键词的得分
        phrase_lst = sorted(phraseScore,key=phraseScore.__getitem__,reverse=True) # 按得分值排序
        '''
        for phrase in phrase_lst[:n]:
            print('(',phrase,',', phraseScore[phrase],')')
        '''
        print(phrase_lst[:n])

if __name__=="__main__":
    t=Rake()
    filecontent = '今天有传在北京某小区，一光头明星因吸毒被捕的消息。' \
                  '下午北京警方官方微博发布声明通报情况，证实该明星为李代沫。' \
                  '李代沫伙同另外6人，于17日晚在北京朝阳区三里屯某小区的暂住地内吸食毒品，' \
                  '6人全部被警方抓获，且当事人对犯案实施供认不讳。'
    t.extract_keywords(filecontent)