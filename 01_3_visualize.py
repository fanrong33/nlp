# encoding: utf-8
# 可视化词向量

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import PCA
from matplotlib import pyplot

# 1、加载持久化的模型
model = Word2Vec.load('sample.en.text.model')
X = model[model.wv.vocab]
print(X)
'''
[[ 0.00289291 -0.00353439  0.00180173 ..., -0.00392716 -0.00130695
   0.00016637]
 ...,
 [ 0.00325975  0.00309974  0.00310635 ...,  0.00183942 -0.00288429
  -0.00195204]]
'''


# 2、Plot Word Vectors Using PCA
# 我们可以在矢量上训练一个投影方法，比如在scikit-learn中提供的方法，然后使用matplotlib将投影作为散点图进行绘制
pca = PCA(n_components=2)
result = pca.fit_transform(X) # 降维
print(result)
'''
[[ 0.00762296  0.00579156]
 [-0.00641727 -0.00725746]
 ...
 [ 0.02623286  0.00838178]
 [ 0.01062225 -0.0042934 ]]
'''
pyplot.scatter(result[:, 0], result[:, 1])


# 3、用这些词本身来标注图表上的点
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0],result[i, 1]))
pyplot.show()