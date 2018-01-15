# encoding: utf-8

Word2Vec 最著名的效果即是以语义化的方式推断出相似词汇：

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
[('queen', 0.50882536)]
model.doesnt_match("breakfast cereal dinner lunch";.split())
'cereal'
model.similarity('woman', 'man')
0.73723527
model.most_similar(['man'])
[(u'woman', 0.5686948895454407),
 (u'girl', 0.4957364797592163),
 (u'young', 0.4457539916038513),
 (u'luckiest', 0.4420626759529114),
 (u'serpent', 0.42716869711875916),
 (u'girls', 0.42680859565734863),
 (u'smokes', 0.4265017509460449),
 (u'creature', 0.4227582812309265),
 (u'robot', 0.417464017868042),
 (u'mortal', 0.41728296875953674)]




# [深度学习和自然语言处理：诠释词向量的魅力](https://www.jianshu.com/p/6c4ce9ef66aa)