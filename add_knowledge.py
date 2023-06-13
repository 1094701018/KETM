import json
from nltk import pos_tag
import  os
from collections import defaultdict
file='D:\RE2-MODEL\kb.json'
file1='D:\RE2-MODEL\KEAR-main\data\wik_dict.json'
wkdict1=json.load(open(file1, encoding='utf-8'))
print(wkdict1['sing'][0]['senses'][0]['glosses'])
kb1wict=json.load(open(file, encoding='utf-8'))
print(kb1wict['sing'])
# print(xxx)

s=[]
cpet=[]
#file1='D:\RE2-MODEL\RE2\data\scitail/train.txt'
file2='D:\RE2-MODEL\KEAR-main\data\kear\conceptnet.en.csv'
file3='D:\RE2-MODEL\english.txt'
stop=[]
with open(file3,'r',encoding='utf-8') as  f:
    for line in f:
        stop.append(line.rstrip('\n'))



for i in kb1wict.keys():
    #print(i)
    print(kb1wict['person'])
    if i in ['?','a','the','or','e.g.','boys']:
       print(i,kb1wict[i])
    s.append(i)


# with open(file2,'r',encoding='utf-8') as f:
#      for line in f:
#          r, s1,s2, label = line.rstrip().split('\t')
#          cpet.extend([s1,s2])
print(stop)
cpet=set(cpet)
vocab=[]
count1=0
count=0
def load_data(count,count1,split=None):
    data = []
    with open(r'D:\RE2-MODEL\RE2\data\mednli\{}.txt'.format(split),encoding='utf-8') as f, \
            open(r'D:\RE2-MODEL\mednli\{}.txt'.format(split), 'w', encoding='utf-8') as fout:
            for line in f:
               # print(line)
                c=[]
                kb={}
                count=count+1
                print(count)
                text1,text2, label = line.rstrip().split('\t')
                a=text1.split()

                c1=set(a)
                for w in c1:
                   if w not in stop:
                       c.append(w)
                c=set(c)
                words=(c.intersection(s)).intersection(cpet)
                l=len(words)
                print((c.intersection(s)).intersection(cpet))
                if l!=0:
                   count1=count1+1
                   for w in words:
                       kb[w]=kb1wict[w]

                vocab.extend(c)
                print('a:{},b:{},c:{}'.format(text1,text2,label))
                data.append({
                    'text1': text1,
                    'text2': text2,
                    'target': label,
                    'kb':kb
                })
                # data.append({
                # 'text1': text1,
                # 'target': label,
                # 'kb': kb
                # })
               # fout.write('{}\t{}\t{}\t{}\n'.format(text1, text2,kb,label))
                fout.write('{}\t{}\t{}\t{}\n'.format(text1,text2,kb,label))
    return data,vocab,count1,count
data,vocab,count1,count=load_data(count,count1,'dev')
print(vocab)
vocab=list(set(vocab))
print(len(vocab))
print(len(set(vocab).intersection(s)))
print(count1)
print(count)
