# encoding: utf-8
# 词向量

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#inp='lastread.txt'
# 加载数据集
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]

outp1 = 'wiki.zh.text.model'
outp2 = 'wiki.zh.text.vector'

# 训练模型
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.pkl')
# load model
new_model = Word2Vec.load('model.pkl')
print(new_model)





# model.save(outp1)
# model.wv.save_word2vec_format(outp2, binary=False)
# print(model)
