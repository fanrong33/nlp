# encoding: utf-8
# word2vec模型使用方法

from gensim.models import Word2Vec


# 加载模型
model = Word2Vec.load('sample.en.text.model')


# 1、获取词向量
print(model['狂砍'])
type(model['狂砍'])


# 2、计算一个词的最近似的词，倒排序
result = model.most_similar('得到')
for each in result:
    print(each[0], each[1])


# 3、计算两个词之间的余弦相似度
sim1 = model.similarity('得到', '拿下')
sim2 = model.similarity('得到', '狂砍')
sim3 = model.similarity('摘得', '抓下')
sim4 = model.similarity('摘得', '仅得')
print(sim1)
print(sim2)
print(sim3)
print(sim4)


# 4、计算两个集合之间的余弦相似度
list1 = ['得到', '拿下']
list2 = ['送出', '狂送']
list_sim1 = model.n_similarity(list1, list2)
print(list_sim1)


# 5、选出集合中不同类的词语
list = ['得到', '拿下', '库里', '摘得']
print(model.doesnt_match(list))
list = ['送出', '狂送', '抓下', '乔丹']
print(model.doesnt_match(list))
