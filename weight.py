# -*- coding:utf-8 -*-
import jieba
import jieba.posseg as psg
#数据准备
sentence = []
sentence1 = []
with open(r'C:\Users\Administrator\PycharmProjects\chenshiyu\online425_req_split', encoding='UTF-8') as f1:
    for line in f1:
        line=line.strip('\n').split('_')
        sentence.append(line)

with open(r'C:\Users\Administrator\PycharmProjects\chenshiyu\online425_req_split', encoding='UTF-8') as f1:
    for line in f1:
        line=''.join(line.strip('\n').split('_'))
        sentence1.append(line)

#名词(n )或动词(v）0.6，形容词(a)或副词(d)0.3，其他词性0.2
def weight(sentence):
    a = []
    w = []
    for i in sentence:
        words = psg.lcut(str(i))
        cx = list(map(lambda x: list(x)[1], words))
        for j in cx:
            if j == 'n' or j == 'v':
                wt = 0.6
            elif j == 'a' or j == 'd':
                wt = 0.3
            else:
                wt = 0.2
            a.append(wt)
        w.append(a)
    return w


result = weight(sentence1)
print(result[0])
'''

words = psg.lcut(str(sentence1[1]))
cx = list(map(lambda x: list(x)[1], words))
a = []
for j in cx:
    if j == 'n' or j == 'v':
        wt = 0.6
    elif j == 'a' or j == 'd':
        wt = 0.3
    else:
        wt = 0.2
    a.append(wt)

print(a)
'''

