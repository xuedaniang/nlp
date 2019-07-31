# -*- coding:utf-8 -*-
import csv
import numpy as np
import jieba
import jieba.analyse
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import scipy.stats
#（一）数据处理
# 选取问题列+回答列数据
sentence = []
with open('C:/Users/chenshiyu/Desktop/csy_data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        li = [row['question']]
        sentence.append(str(li))
#print(sentence)
#（二）建模

tf_vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(tf_vectorizer.fit_transform(sentence))


lda = LatentDirichletAllocation(n_topics=100,
                                learning_offset=50.,
                                random_state=0)

dscore=lda.fit_transform(tfidf)
p=np.array(dscore[17])
q=np.array(dscore[18])
M=(p+q)/2
print(0.5*np.sum(p*np.log(p/M))+0.5*np.sum(q*np.log(q/M)))

estimator = KMeans(n_clusters=100)
estimator.fit(dscore)
print(estimator.labels_)

print(metrics.silhouette_score(dscore, estimator.labels_, metric='euclidean'))

#for i in range(len(sentence)):
    #print('问题：'+str(sentence[i]),'类别：'+str(estimator.labels_[i]))

for i in range(len(sentence)):
    if estimator.labels_[i]==1:
        print(sentence[i])




