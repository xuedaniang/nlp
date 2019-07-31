
# -*- coding:utf-8 -*-
import pickle
import pandas as pd
import random
import csv
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
#数据准备
# 读取训练集

def readtrain():
    with open(r'C:\Users\chenshiyu\PycharmProjects\first\label_data.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        column = [row for row in reader]
    content = [i[1] for i in column[1:]] #第一列为文本内容，并去除列名
    label = [i[2] for i in column[1:]] #第二列为类别，并去除列名
    train = [content, label]
    return train

# 对列表进行分词并用空格连接
def segmentWord(cont):
    c = []
    for line in cont:
        line = line.strip('\n').split('_')
        c.append(line)
    return c

data = readtrain()
content = segmentWord(data[0])
label = data[1]

# 划分
train_content = content[:1800]
test_content = content[1800:]
train_label = label[:1800]
test_label = label[1800:]
#tfidf向量
with open(r'C:\Users\chenshiyu\PycharmProjects\first\tfidf\tfidf_model', 'rb') as model_fl:
    ti_model = pickle.load(model_fl, encoding="iso-8859-1")
with open(r'C:\Users\chenshiyu\PycharmProjects\first\tfidf\tfidf_dict', 'rb') as dict_fl:
    ti_dict = pickle.load(dict_fl, encoding="iso-8859-1")
vector = []
for line in content:
    bow = ti_dict.doc2bow(line)
    vec = [(ti_dict[w], f) for w, f in ti_model[bow]]
    vector.append(vec)
print(len(vector))
#将词向量转化为句子向量w2v
w2c_model = Word2Vec.load(r'C:\Users\chenshiyu\PycharmProjects\first\w2c_model_20180717\w2c_model')
#利用tfidf权重对w2v词向量进行加权平均
def v(vector):
    ff=[]
    v1=[]
    nsv = []
    for i in vector:
        for w, f in i:
            ff.append(f)
            if w in w2c_model.wv.vocab:
                wv=w2c_model[w]
            else:
                wv=[0.0] * 100
            v1.append(f * np.array(wv))
        q = np.array(v1).sum(axis=0) / sum(ff)
        nsv.append(q)
    return nsv
nsv=v(vector)
print(len(nsv))
train_vector =nsv[:1800]
test_vector = nsv[1800:]

#print(len(test_vector))
'''
def transform(a):
    vector = []
    nvector = []
    for line in a:
        bow = ti_dict.doc2bow(line)
        vec = [(ti_dict[w], f) for w, f in ti_model[bow]]
        weight = [v[1] for v in vec]
        vector.append(weight)
        for line in vector:
            nline = [0] * (18 - len(line)) + line
        nvector.append(nline)
    return nvector

train_vector = transform(train_content)
test_vector = transform(test_content)
'''
#print(len(train_vector))
'''
label=[]
for i in train_label:
    s=int(i)
    label.append(s)
print(label)
'''
#print(len(test_vector))


knn = neighbors.KNeighborsClassifier()
knn_fit=knn.fit(train_vector, train_label)
predicted = knn.predict(test_vector)
print('knn',accuracy_score(predicted, test_label))
#print(classification_report(predicted, test_label))





# 训练和预测一体
'''
NB = MultinomialNB()
NB_fit = NB.fit(train_vector, train_label)
predicted = NB.predict(test_vector)
print('贝叶斯',accuracy_score(predicted, test_label))
#print(classification_report(predicted, test_label))

SV=SVC()
SV_fit = SV.fit(train_vector, train_label)
predicted = SV.predict(test_vector)
print('SVM',accuracy_score(predicted, test_label))
#print(classification_report(predicted, test_label))

knn = neighbors.KNeighborsClassifier()
knn_fit=knn.fit(train_vector, train_label)
predicted = knn.predict(test_vector)
print('knn',accuracy_score(predicted, test_label))
#print(classification_report(predicted, test_label))

rf=RandomForestClassifier()
rf_fit=rf.fit(train_vector, train_label)
predicted = rf.predict(test_vector)
print('rf',accuracy_score(predicted, test_label))

AdB=AdaBoostClassifier()
AdB_fit=AdB.fit(train_vector, train_label)
predicted = AdB.predict(test_vector)
print('AdaBoost',accuracy_score(predicted, test_label))


nn=MLPClassifier(max_iter=10000)
nn_fit=nn.fit(train_vector, train_label)
predicted = nn.predict(test_vector)
print('nn',accuracy_score(predicted, test_label))
'''