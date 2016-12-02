# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:48:18 2015

@author: miaomiao
"""

import os
import numpy
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn import metrics 
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') 

def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred);
    m_recall = metrics.recall_score(actual,pred);
    print 'predict info:'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall);
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred));

base=open('D:/test/no/vectors.bin','r')
embeddings={}
for baseline in base:
    item=baseline.split(' ')
    word=item[0]
    '''
    temp=item[0].split('_')
    if len(temp)<=1:
        word = temp[0]
    else:
        word = temp[0]+'_'+temp[1].lower()  
    '''
    embeddings[str(word)]=numpy.asarray([float(i) for i in item[1:-1]])
    #print word,'~~~~~~~~~',embeddings[str(word)]  #小写
    #embeddings[str(word)]=baseline
base.close()

documents=[]
label=[]
test_docs=[]
test_labels=[]
train_base = 'D:/test/no/train/'
train_current_files = os.listdir('D:/test/no/train/')
for i in range(2):
    train_contents=[]
    sub_dir = train_base+'//'+train_current_files[i]
    train_current_file = os.listdir(sub_dir)
    for filename in train_current_file:
        filename = sub_dir+'//'+filename
        f = open(filename,'r')
        train_content = f.read()
        train_content = train_content.decode('utf8','ignore')
        train_contents.append(train_content)
        f.close()
test_base = 'D:/test/no/test/'
test_current_files = os.listdir('D:/test/no/test/')
for i in range(2):
    test_contents=[]
    sub_dir = test_base+'//'+test_current_files[i]
    test_current_file = os.listdir(sub_dir)
    for filename in test_current_file:
        filename = sub_dir+'//'+filename
        e = open(filename,'r')
        test_content = e.read()
        test_content = test_content.decode('utf8','ignore')
        test_contents.append(test_content)
        e.close()
    #sougou_train=contents[0:6000]
    #sougou_test=contents[6000:8000]

    count_v1= CountVectorizer(stop_words = 'english')  #将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在文本i下的词频
    counts_train = count_v1.fit_transform(train_contents)  #fit_transform将文本转为词频矩阵 
    counts_test = count_v1.transform(test_contents)  

    tfidftransformer = TfidfTransformer()  #统计每个词语的tf-idf权值
    tfidf_train = tfidftransformer.fit_transform(counts_train)  #fit_transform计算tf-idf
    tfidf_test = tfidftransformer.transform(counts_test)

    words=count_v1.get_feature_names()

    for j in range(2):
        document1 = tfidf_train[j]                #每个document处理一个文档将值返回到一个统一的documents中，再进行分类处理
        document1 = document1.todense().tolist()[0]
        
        document1_txt=numpy.asarray([0]*100,dtype='float')
        document1_content=re.split(' ',train_contents[j])
        #print document1_content  #大写
        current_embeddings1=[]
        current_word1=[]
        for index,word in enumerate(document1_content):   
            #print word #大写
            #print words[index]  #小写
            if embeddings.has_key(word):
                
                current_embeddings1.append(embeddings[word])
                current_word1.append(word)
                #print current_word1
                pass
            pass
        temp_embeddings1=current_embeddings1
        for e in range(len(current_embeddings1)):
            for f in range(100):
                if e==0:
                    temp_embeddings1[e][f] = max(current_embeddings1[0][f],current_embeddings1[1][f])
                elif e==len(current_embeddings1)-1:
                    temp_embeddings1[e][f] = max(current_embeddings1[e-1][f],current_embeddings1[e][f])
                else:
                    temp_embeddings1[e][f] = max(current_embeddings1[e][f],current_embeddings1[e-1][f],current_embeddings1[e+1][f])
 
        for index,val in enumerate(document1):
            
            if val!=0.0:
                #print '!!!!!!!!!!!!!',words[index],'!!!!!!!!!!!!!',current_word1
                temp=words[index].split('_')
                if len(temp)<=1:
                    temp = temp[0]
                else:
                    temp = temp[0]+'_'+temp[1].upper() #tolower大写变小写，embeddings.has_key(大写)，words[index](小写)
                if embeddings.has_key(temp):
                    #print words[index],'~~~~',embeddings[words[index]],'\n'
                    m=0
                    if not temp==current_word1[m] :
                        m=m+1
                    else:
                        continue
                    document1_txt=document1_txt+temp_embeddings1[m]*val
                else: 
                    continue
            else:
                continue
        if len(current_embeddings1)==0:
            continue
        else:
             
            documents.append(document1_txt)
            label.append(i)
    
    for j in range(2):
        document2 = tfidf_test[j]                #每个document处理一个文档将值返回到一个统一的documents中，再进行分类处理
        document2 = document2.todense().tolist()[0]
        document2_txt=numpy.asarray([0]*100,dtype='float')
        document2_content=re.split(' ',test_contents[j])
        current_embeddings2=[]
        current_word2=[]
        for index,word in enumerate(document2_content):
            if embeddings.has_key(word):
                
                current_embeddings2.append(embeddings[word])
                current_word2.append(word)
                pass
            pass
        temp_embeddings2=current_embeddings2
        for e in range(len(current_embeddings2)):
            for f in range(100):
                if e==0:        
                    temp_embeddings2[e][f] = max(current_embeddings2[0][f],current_embeddings2[1][f])
                elif e==len(current_embeddings2)-1:
                    temp_embeddings2[e][f] = max(current_embeddings2[e-1][f],current_embeddings2[e][f])
                else:
                    temp_embeddings2[e][f] = max(current_embeddings2[e][f],current_embeddings2[e-1][f],current_embeddings2[e+1][f])
 
        for index,val in enumerate(document2):
            if val!=0.0:
                temp=words[index].split('_')
                if len(temp)<=1:
                    temp = temp[0]
                else:
                    temp = temp[0]+'_'+temp[1].upper() #tolower大写变小写，embeddings.has_key(大写)，words[index](小写)
                if embeddings.has_key(temp):
                    #print words[index],'~~~~',embeddings[words[index]],'\n'
                    m=0
                    if not temp==current_word2[m] :
                        m=m+1
                    else:
                        continue
                    document2_txt=document2_txt+temp_embeddings2[m]*val
                else: 
                    continue
            else:
                continue
        if len(current_embeddings2)==0:
            continue
        else:
            test_docs.append(document2_txt)
            test_labels.append(i)  
       
from sklearn.svm import SVC  
print '*************************\nTrain\n*************************'  
svc = SVC() 
svc.fit(documents,label)
pred = svc.predict(documents)
calculate_result(label,pred)
print '*************************\nTest\n*************************'
pred = svc.predict(test_docs)
calculate_result(test_labels,pred)