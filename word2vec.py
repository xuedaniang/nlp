# -*- coding:utf-8 -*-
import csv
'''
with open(r'F:\sj.csv', 'w', encoding='UTF-8') as f2:
    f2.writelines(rows)
    f2.write('\n')
  '''
sentence1 = []
with open(r'C:\Users\Administrator\PycharmProjects\chenshiyu\online425_req_split', encoding='UTF-8') as f1:
    for line in f1:
        line=''.join(line.strip('\n').split('_'))
        sentence1.append(line)

with open(r'F:\sj.csv', 'w', encoding='UTF-8') as f2:
    writer=csv.writer(f2)
    for i in sentence1:
        writer.writerow(i)

