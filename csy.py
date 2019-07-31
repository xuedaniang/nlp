# -*- coding:utf-8 -*-

import csv
import numpy as np
import jieba
import jieba.analyse
import gensim
from gensim.models import Word2Vec
from gensim import matutils
import sklearn
from sklearn.cluster import KMeans
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import Birch
import nltk
from nltk.cluster import KMeansClusterer
from nltk.cluster.gaac import GAAClusterer
#（一）数据处理
# 选取问题列数据
sentence = []
with open('C:/Users/chenshiyu/Desktop/csy_data.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        li = jieba.lcut(row['question'])
        li.strip('\n').split('_')
        sentence.append(li)
print(sentence)


'''
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords
stopwords=stopwordslist('C:/Users/chenshiyu/Desktop/stopword.txt')
print(stopwords)
#去除停用词
def removestopwords(list):
    outstr=[]
    for l in list:
        for word in l:
            if word not in stopwords:
                outstr.append(word)
    return outstr

out=removestopwords(list)
print(out)
'''
# 提取关键词
keywords = jieba.analyse.extract_tags(str(sentence), topK=100, withWeight=True, allowPOS=())
for item in keywords:
    print(item[0], item[1])   # 分别为关键词和相应的权重


# 建立word2vector模型
model = Word2Vec(sentence, size=100, window=5, min_count=5, workers=4)
outml = 'w2v.model'
model.save(outml)
model = Word2Vec.load(outml)
print(model)


# 获取词向量
print(model[u'帮主'])
#计算一个词的近似词
result = model.most_similar('傻')
for each in result:
   print(each[0], each[1])
#计算两词的相似度
sim1 = model.similarity(u'帮主', u'帅')
print(('帮主，帅'),sim1)

#加载新的w2v模型
w2c_model = Word2Vec.load('C:/Users/chenshiyu/Desktop/model_w2c_20180522')

#将词向量转化为句子向量
def text_vec(tx_list):
    vec = []
    for w in tx_list:
        if w in w2c_model.wv.vocab:
            vec.append(w2c_model.wv[w])
        else:
            vec.append([0.0] * 100)
    return list(matutils.unitvec(array(vec).mean(axis=0)))


sv = []
for i in sentence:
    try:
        s = text_vec(i)
        sv.append(s)
    except Exception as ex:
        print(ex)
        continue
#print(sv[0:3])


#（2）聚类\聚类质量
'''
#利用轮廓系数选择k
def k(nsv):
    scores = []
    ki = []
    for k in range(100):
        estimator = KMeans(n_clusters=k)
        estimator.fit(nsv)
        scores.append(metrics.silhouette_score(nsv, estimator.labels_, metric='euclidean'))
        rs = sorted(scores, reverse=True)
        for i in rs[:10]:
            l = scores.index(i)
            ki.append(l)
    return ki

print(k(nsv))

'''
#轮廓系数:lk=metrics.silhouette_score(sv, labels, metric='euclidean')

# Calinski-Harabaz 指数:ch=metrics.calinski_harabaz_score(sv, labels)
'''
#K均值聚类
'利用SSE选择k'
SSE = []  # 存放每次结果的误差平方和
for k in range(5, 30):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(sv)
    SSE.append(estimator.inertia_)
X = range(5, 30)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()

#利用轮廓系数选择k

scores=[]
for k in range(100):
    estimator = KMeans(n_clusters=k)
    estimator.fit(sv)
    scores.append(metrics.silhouette_score(sv, estimator.labels_, metric='euclidean'))
    print(scores.index(max(scores))

x = range(5,200)
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.plot(x, scores, 'o-')
plt.show()


km = KMeans(n_clusters=15)  # 聚类算法，参数n_clusters=15，聚成15类
y_pred = km.fit_predict(sv)  # 直接对数据进行聚类，聚类不需要进行预测
print('Predicting result:',y_pred)
cluster = km.labels_.tolist()
print(cluster)
c=pd.Series(km.labels_).value_counts()
print(c)

for i in range(15):
    ci=cluster.count(i)
    print(ci/len(cluster))

cents = km.cluster_centers_#质心
labels = km.labels_#样本点被分配到的簇的索引

#Birch
birch = Birch(n_clusters=None)
# 尝试多个threshold取值，和多个branching_factor取值
param_grid = {'threshold': [0.8, 0.5, 0.3, 0.1], 'branching_factor': [100, 50, 20]}  # 定义优化参数字典，字典中的key值必须是分类算法的函数的参数名
for threshold in param_grid['threshold']:
    for branching_factor in param_grid['branching_factor']:
        clf = Birch(n_clusters=15, threshold=threshold, branching_factor=branching_factor)
        clf.fit(sv)
        y_pred = clf.predict(sv)
        print(threshold, branching_factor, "轮廓系数:", metrics.silhouette_score(sv, y_pred))

km=KMeansClusterer(num_means=15, distance=nltk.cluster.util.cosine_distance)
V=np.array(sv)
pv=km.cluster(V)

klist=[]
for i in V:
    klist.append(km.classify(i))
print(klist)
print(metrics.silhouette_score(sv, klist, metric='euclidean'))


#利用轮廓系数选择k

scores=[]
for k in range(5, 200):
    estimator = sklearn.cluster.MiniBatchKMeans(n_clusters=k)
    estimator.fit(sv)
    scores.append(metrics.silhouette_score(sv, estimator.labels_, metric='euclidean'))

x = range(5,200)
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.plot(x, scores, 'o-')
plt.show()

param_grid = {'damping': [0.8, 0.6, 0.7, 0.5]}
for damping in param_grid['damping']:
        clf = sklearn.cluster.AffinityPropagation(damping=damping)
        clf.fit(sv)
        y_pred = clf.predict(sv)
        print(damping, "轮廓系数:", metrics.silhouette_score(sv, y_pred))
'''
scores=[]
for k in range(5, 20):
    estimator = sklearn.cluster.AgglomerativeClustering(n_clusters=k)
    estimator.fit(sv)
    scores.append(metrics.silhouette_score(sv, estimator.labels_, metric='euclidean'))
x = range(5,20)
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.plot(x, scores, 'o-')
plt.show()

