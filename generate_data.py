from nltk.tokenize import TweetTokenizer
import  re
import json
from nltk import tokenize
from tqdm import tqdm
tokenizer = TweetTokenizer()
def tokenize2(string):
    string = ' '.join(tokenizer.tokenize(string))
    string = re.sub(r"[-.#\"/]", " ", string)
    string = re.sub(r"\'(?!(s|m|ve|t|re|d|ll)( |$))", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
file2='D:\RE2-MODEL\KEAR-main\data\kear\conceptnet.en.csv'
cpet=[]
with open(file2, 'r', encoding='utf-8') as f:
    for line in f:
        r, s1, s2, label = line.rstrip().split('\t')
        cpet.append([s1, s2])


datadir='MEDNLI'
flag=0
for split in ['train','test','dev']:
    file = 'D:\RE2-MODEL\{}\{}.txt'.format(datadir,split)
    file1 = 'D:\RE2-MODEL\dataset\{}1\{}.txt'.format(datadir,split)
    with open(file, 'r', encoding='utf-8') as f, \
            open(file1, 'w', encoding='utf-8') as fout:
        count = 0
        n_lines = 0
        n_lines1 = 0
        print(file)
        # for _ in f:
        #     n_lines=n_lines+1


        for line in f:
            n_lines1 = n_lines1 + 1


            text11, text22, kbstr, label = line.rstrip().split('\t')
          #  text11, kbstr, label = line.rstrip().split('\t')
            text1=tokenize2(text11).split()
            text2=tokenize2(text22).split()

           # print(n_lines)
            # print(text1)
            # print(n_lines)
            kbstr=kbstr.strip()
          #  print(kbstr)
            kb=eval(kbstr)

            kbs =list(kb.keys())


            lens=len(kbs)
            a = ''
            for i in range(len(text1)):
                a = a + text1[i]
                a = a + ' '
            b = ''
            for i in range(len(text2)):
                b = b + text2[i]
                b = b+ ' '

            # print(kbs[0])
            # print(kbs[1]

            if lens==0:
                count=count+1
                print(n_lines1)
              #  fout.write('{}\t{}\t{}\n'.format(a, b, label))
                continue
            else:
                for word in kbs:
                    if word in text1:
                        charu=word+' '+kb[word].replace('\n','')
                        a=a+charu+' '

                    if word in text2:
                        charu = word + ' ' + kb[word].replace('\n','')
                        b = b + charu + ' '
                if a=='' or b=='':
                    c=a+b
                    a=c
                    b=c


            fout.write('{}\t{}\t{}\t{}\t{}\n'.format(text11,a,text22,b, label))


        print(count,n_lines1)