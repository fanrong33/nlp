# encoding: utf-8
# 采用TF-IDF方法提取文本关键词

import sys,codecs
import pandas as pd
import numpy as np
import jieba.posseg
import jieba.analyse
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer

"""
   TF-IDF权重：
       1、CountVectorizer 构建词频矩阵
       2、TfidfTransformer 构建tfidf权值计算
       3、文本的关键字
       4、对应的tfidf矩阵
"""

def data_prepos(text, stopwords):
    '''数据预处理'''

    array = []
    pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']
    seg_list = jieba.posseg.cut(text)
    for i in seg_list:
        if i.word not in stopwords and i.flag in pos: # 去停用词 + 词性筛选
            array.append(i.word)
    return array

def get_keywords_tfidf(data, stopwords, topK):
    ''' tf-idf获取文本top10关键词 '''

    id_list, title_list, abstract_list = data['id'], data['title'], data['abstract']
    corpus = [] # 将所有文档输出到一个list中，一行就是一个文档
    for i in range(len(id_list)):
        text = '%s。%s' % (title_list[i], abstract_list[i]) # 拼接标题和摘要
        text = data_prepos(text, stopwords) # 文本预处理
        text = ' '.join(text) # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    # 2、统计每个词的tf-idf权值
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频

    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        print("输出第%s篇文本的词语tf-idf" % str(i+1))
        ids.append(id_list[i])
        titles.append(title_list[i])
        df_word, df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print(word[j], weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])

        exit()
        df_word = pd.DataFrame(df_word, columns=['word'])
        df_weight = pd.DataFrame(df_weight, columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by='weight', ascending=False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列表并转换成数组格式
        
        word_split = [keyword[x] for x in range(0, topK)] # 抽取前topK个词汇作为关键词
        word_split = ' '.join(word_split)
        keys.append(word_split)

    result = pd.DataFrame({'id': ids, 'title': titles, 'key': keys}, columns=['id', 'title', 'key'])
    return result


if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv('data/sample_data.csv')
    # 停用词表
    stopwords = [w.strip() for w in codecs.open('data/stopword.txt', 'r', encoding='utf-8').readlines()]
    # tf-idf关键词提取
    result = get_keywords_tfidf(data, stopwords, 10)
    result.to_csv('result/keys_TFIDF.csv', index=False)



