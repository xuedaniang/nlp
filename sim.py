# -*- coding:utf-8 -*-
import scipy
from scipy.stats import zscore
import pickle
import numpy as np
import gensim
from gensim.models import Word2Vec
import csv
from gensim import matutils
import sklearn
from sklearn.cluster import KMeans
from numpy import array
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as dist
from sklearn.preprocessing import scale
from math import tanh

#数据准备
sentence = []
with open(r'C:\Users\Administrator\PycharmProjects\chenshiyu\online425_req_split', encoding='UTF-8') as f1:
    for line in f1:
        line=line.strip('\n').split('_')
        sentence.append(line)
#for line in sentence:
   # print(line)

w2c_model = Word2Vec.load(r'C:\Users\chenshiyu\PycharmProjects\first\w2c_model_20180717\w2c_model')
#将词向量转化为句子向量w2v

for line in sentence:
    for w in line:
        if w in w2c_model.wv.vocab:
            wvec=w2c_model.wv[w]
        else:
            wvec=[0.0] * 100

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

#tfidf向量
with open(r'C:\Users\chenshiyu\PycharmProjects\first\tfidf\tfidf_model', 'rb') as model_fl:
    ti_model = pickle.load(model_fl, encoding="iso-8859-1")
with open(r'C:\Users\chenshiyu\PycharmProjects\first\tfidf\tfidf_dict', 'rb') as dict_fl:
    ti_dict = pickle.load(dict_fl, encoding="iso-8859-1")
'''
vector = []
for line in sentence:
    bow = ti_dict.doc2bow(line)
    for w in line:
        if w in ti_dict:
            vec = [(ti_dict[w], f) for w, f in ti_model[bow]]
            vector.append([v[1] for v in vec])
        else:
            vector.append([0.0] * len(line))
            '''
vector = []
for line in sentence:
    bow = ti_dict.doc2bow(line)
    vec = [(ti_dict[w], f) for w, f in ti_model[bow]]
    vector.append(vec)
for i in vector[:10]:
    print(i)

#for line in vector:
    #print(line)
#print(len(vector))
#print(len(sentence))
'''
#利用tfidf权重对w2v词向量进行加权平均
def v(vector):
    ff=[]
    v1=[]
    nsv = []
    for i in vector:
        for w, f in i:
            ff.append(f)
            if w in w2c_model.wv.vocab:
                wv = w2c_model[w]
            else:
                wv = [0.0] * 100
            v1.append(f * np.array(wv))
        q = np.array(v1).sum(axis=0) / sum(ff)
        nsv.append(q)
    return nsv

nsv=v(vector)

#print(len(nsv))
'''
#相似度算法1
def sim1(vec1,vec2):
    s=[]
    for w1, f1 in vec1:
        for w2, f2 in vec2:
            try:
                sim = f1 * f2 * tanh(w2c_model.similarity(w1, w2) - 0.5)
                s.append(sim)
            except KeyError as ke:
                print(ke)

    return tanh(sum(s))


#相似度算法2
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def sim2(sent1,sent2):
    a=[]
    b=[]
    n1 = len(sent1)
    n2 = len(sent2)
    global sm
    for i in sent1:
        for j in sent2:
            try:
                sim = tanh(w2c_model.similarity(i, j)-0.5)
                a.append(sim)
                aa = [a[i:i + n2] for i in range(0, len(a), n2)]
                for k in aa:
                    maxs = max(k)
                    b.append(maxs)
                    sm = (sum(b)) /n1
            except KeyError as ke:
                pass
    return sm

def sim3(sent1,sent2)  :
    s=(sim2(sent1,sent2)+sim2(sent2,sent1))/2
    return tanh(s)



'''

def sample1(n):
    for i in range(0, len(sentence), n):
        s = sim1(vector[i], vector[2040])
        print([sentence[i], sentence[2040]], s)

def sample2(n):
    for i in range(0, len(sentence), n):
        s = sim2(sentence[i], sentence[2040])
        print([sentence[i], sentence[2040]],s)
#sample1(100)
print('~'*50)
sample2(100)
#sample(1000)
'''
#载入测试集
with open(r'C:\Users\chenshiyu\PycharmProjects\first\test_data1.csv') as testdata:
    reader = csv.DictReader(testdata)
    origin=[]
    compare=[]
    testgrade=[]
    for row in reader:
        origin .append(row['origin'].strip('\n').split('_'))
        compare.append(row['compare'].strip('\n').split('_'))
        testgrade.append(row['grade'])
        testgrade = [int(i) for i in testgrade]
def simgrade(origin,compare):
    traingrade=[]
    for a,b in zip(origin,compare):
        grade=sim3(a,b)
        traingrade.append(grade)
    return traingrade
traingrade=simgrade(origin,compare)


def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[-half]) / 2
def diff(testgrade,traingrade):
    sumdiff=[]
    for a,b in zip(testgrade,traingrade):
        diff=float(a)-float(b)
        sumdiff.append(abs(diff))
    return sum(sumdiff)/len(testgrade),get_median(sumdiff),sum(map(lambda x: x < 0.1, sumdiff))


result=diff(testgrade,traingrade)

#print(result)

#print(traingrade)
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
traingrade=np.array(traingrade)
traingrade=list(np.where(traingrade>0,1,-1))
#print(traingrade)
print('准确率:',accuracy_score(testgrade,traingrade),'召回率:',recall_score(testgrade,traingrade),'F1值:',f1_score(testgrade,traingrade))
