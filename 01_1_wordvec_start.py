# encoding: utf-8
# word2vec的模型是基于神经网络来训练词向量模型；
# word2vec的主要的应用还是自然语言的处理，通过训练出来的词向量，可以进行聚类等处理，或者作为其他深入学习的输入。


from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# 1、加载句子集合
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]

# 2、训练模型
model = Word2Vec(sentences, min_count=1)
# PS：sentences 有不同的处理类可以使用，方便从文件加载处理为句子集合

print(model)
'''  Word2Vec(vocab=14, size=100, alpha=0.025) '''

# 总结词汇
words = list(model.wv.vocab)
print(words)
''' ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec', 'second', 'yet', 'another', 'one', 'more', 'and', 'final'] '''

# 查看某个词的词向量
print(model['first'])
'''
[-0.00250688 -0.00162008  0.00250901 -0.00097651 -0.0048413  -0.00068062
  0.00435694 -0.00375766 -0.00431498  0.0019443  -0.00128378 -0.00343237
  ...
 -0.00034512  0.00334434  0.00412283  0.00035618 -0.00126278 -0.00482794
  0.00386906 -0.00355957 -0.00194974 -0.00251286]
'''
print(len(model['first']))
''' 100  词向量为100维度 '''

# 所以，则为所有的词的向量集合，理解word2vec的结构！
print(model[model.wv.vocab])
'''
[[  3.35454009e-03  -2.96757789e-03   8.95642443e-04 ...,   4.16836003e-03
   -3.26405023e-03  -1.91481831e-03]
 ...,
 [  7.19302261e-05   1.70022575e-03   3.59526509e-03 ...,   1.11010019e-03
    3.70053225e-03  -3.61868995e-03]]
'''

# 3、持久化模型
model.save('sample.en.text.model')
model.wv.save_word2vec_format('sample.en.text.vector', binary=True)
'''
save() 函数保存的完整的模型？额
wv.save_word2vec_format() 函数保存的其实就是词汇和对应向量，不过会丢失tree信息，所以无法进行增量训练
'''


# 4、加载持久化的模型，需与上面持久化的模型对应，此为方法一
new_model = Word2Vec.load('sample.en.text.model')
print(new_model)

# 4、加载持久化模型，方法二
from gensim.models import KeyedVectors

filename = 'sample.en.text.vector'
new_model = KeyedVectors.load_word2vec_format(filename, binary=True)


# 参考：
# [word2vec学习小记](https://www.jianshu.com/p/418f27df3968)
# [How to Develop Word Embeddings in Python with Gensim](https://machinelearningmastery.com/develop-word-embeddings-python-gensim/)
# [gensim.model.word2vec API](https://radimrehurek.com/gensim/models/word2vec.html)
