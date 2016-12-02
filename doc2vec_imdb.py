# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:01:11 2016

@author: wm
"""

import gensim
import os
from random import shuffle
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from sklearn.cross_validation import train_test_split
import numpy as np

pos_reviews=[]
neg_reviews=[]
unsup_reviews=[]
count1=0
count2=0
count3=0
pos_files_train= os.listdir('aclImdb/train/pos')
for filename in pos_files_train:
    filename = 'aclImdb/train/pos/'+filename
    with open(filename,'r') as infile:
        string=('').join(infile.readlines())
        pos_reviews.append(string)
    count1+=1
    if count1==1000:
        break

neg_files_train= os.listdir('aclImdb/train/neg')
for filename in neg_files_train:
    filename = 'aclImdb/train/neg/'+filename
    with open(filename,'r') as infile:
        string=('').join(infile.readlines())
        neg_reviews.append(string)
    count2+=1
    if count2==1000:
        break

unsup_files_train= os.listdir('aclImdb/train/unsup')
for filename in unsup_files_train:
    filename = 'aclImdb/train/unsup/'+filename
    with open(filename,'r') as infile:
        string=('').join(infile.readlines())
        unsup_reviews.append(string)
    count3+=1
    if count3==1000:
        break

#use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)

#Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
#We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
#a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i,v in enumerate(reviews):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

size = 400

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=10)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=10)

#build vocab over all reviews
train_corpus_vob  = x_train+ x_test+ unsup_reviews
model_dm.build_vocab(train_corpus_vob)
model_dbow.build_vocab(train_corpus_vob)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
all_train_reviews = np.concatenate((x_train, unsup_reviews))
all_train_reviews = x_train+ unsup_reviews
for epoch in range(15):
    shuffle(all_train_reviews)
    model_dm.train(all_train_reviews)
    model_dbow.train(all_train_reviews)

#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

#train_vecs_dm = getVecs(model_dm, x_train, size)
#train_vecs_dbow = getVecs(model_dbow, x_train, size)


train_vecs_dm = np.concatenate([model_dm.infer_vector(x_train[i].words).reshape((1,size)) for i in range(len(x_train))])
train_vecs_dbow = np.concatenate([model_dbow.infer_vector(x_train[i].words).reshape((1,size)) for i in range(len(x_train))])


train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

#train over test set
#x_test = np.array(x_test)

#for epoch in range(10):
#    shuffle(x_test)
#    model_dm.train(x_test)
#    model_dbow.train(x_test)

test_vecs_dm = np.concatenate([model_dm.infer_vector(x_test[i].words).reshape((1,size)) for i in range(len(x_test))])
test_vecs_dbow = np.concatenate([model_dbow.infer_vector(x_test[i].words).reshape((1,size)) for i in range(len(x_test))])


##Construct vectors for test reviews
#test_vecs_dm = getVecs(model_dm, x_test, size)
#test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD  
from keras.objectives import binary_crossentropy,mean_squared_error
from keras.regularizers import l2
print 'start !'
model = Sequential()  
model.add(Dense(output_dim=400,input_dim=800,W_regularizer=l2(0.001)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))  
model.add(Dense(output_dim=2,W_regularizer=l2(0.001)))  
model.add(Activation('softmax'))

from keras.utils import np_utils
nb_classes = 2
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
 
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),metrics=['accuracy'])   
#Begin train,nb_epoch:epoch number  
model.fit(train_vecs, y_train, nb_epoch=100, batch_size=16,shuffle=True,verbose=2,show_accuracy=True,validation_data=(test_vecs, y_test))
model.evaluate(test_vecs, y_test, batch_size=16) 
predict_labels=model.predict_classes(test_vecs)
predict_labels_train=model.predict_classes(train_vecs)

print('\n'+'Keras Test Accuracy:'+str(float(len([1 for i,j in zip(y_test,predict_labels) if i[j]==1.0]))/len(y_test)))
print('\n'+'Keras Train Accuracy:'+str(float(len([1 for i,j in zip(y_train,predict_labels_train) if i[j]==1.0]))/len(y_train)))