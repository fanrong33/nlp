# encoding: utf-8
# 增量训练，如果可能的话，把所有的例子合并成一个语料库，做一个大的词汇 - 然后训练。

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


# 1、加载持久化的模型
model = Word2Vec.load('sample.en.text.model')


# 2、增量训练句子集
sentences = [['i', 'like', 'jinjiang']]
print(model.corpus_count)
print(model.iter)
model.build_vocab(sentences, update=True)
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

print(list(model.wv.vocab))


# 3、持久化模型
model.save('sample.en.text.model')
model.wv.save_word2vec_format('sample.en.text.vector', binary=True)




# [Incremental Word2Vec Model Training in gensim](https://stackoverflow.com/questions/42746007/incremental-word2vec-model-training-in-gensim)