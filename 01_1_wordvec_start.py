# encoding: utf-8
# 词向量

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

# 所以，则为所有的词的向量集合
print(model[model.wv.vocab])


# 3、持久化模型
model.save('sample.en.text.model')
model.wv.save_word2vec_format('sample.en.text.vector', binary=True)
'''
save() 函数保存的完整的模型？额
wv.save_word2vec_format() 函数保存的其实就是词汇和对应向量，不过会丢失tree信息，所以无法进行增量训练
'''


# 4、加载持久化的模型
new_model = Word2Vec.load('sample.en.text.model')
print(new_model)


