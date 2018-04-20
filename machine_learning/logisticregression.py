#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:14:42 2018

@author: carl
"""

import numpy as np
np.set_printoptions(suppress=True)
from numpy import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
stemmer =  LancasterStemmer()
lemmer = WordNetLemmatizer()
#print(stemmer.stem('dictionaries'))
#print(lemmer.lemmatize('dictionaries'))

from gensim import models
import numpy as np
from pandas import DataFrame, Series
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from gensim import models
import matplotlib.pyplot as plt
import seaborn

from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
stop = stopwords.words('english')
alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')

from subprocess import check_output
print(check_output(["ls", "/Users/carl/Desktop/UWO/Machine Learning/project"]).decode("utf8"))

df_train = pandas.read_csv('/Users/carl/Desktop/UWO/Machine Learning/project/train.csv').dropna()
texts = np.concatenate([df_train.question1.values, df_train.question2.values])

def process_sent(words, lemmatize=False, stem=False):
    words = words.lower()
    tokens = alpha_tokenizer.tokenize(words)
    for index, word in enumerate(tokens):
        if lemmatize:
            tokens[index] = lemmer.lemmatize(word)
        elif stem:
            tokens[index] = stemmer.stem(word)
        else:
            tokens[index] = word
    return tokens

corpus_lemmatized = [process_sent(sent, lemmatize=True, stem=False) for sent in texts]
corpus_stemmed = [process_sent(sent, lemmatize=False, stem=True) for sent in texts]
corpus = [process_sent(sent) for sent in texts]

VECTOR_SIZE = 100
min_count = 10
size = VECTOR_SIZE
window = 10

model_lemmatized = models.Word2Vec(corpus_lemmatized, min_count=min_count, 
                                   size=size, window=window)
model_stemmed = models.Word2Vec(corpus_stemmed, min_count=min_count, 
                                size=size, window=window)
model = models.Word2Vec(corpus, min_count=min_count, 
                                size=size, window=window)

Y = np.array(df_train.is_duplicate.values)[200000:]

def preprocess_check(words, lemmatize=False, stem=False):
    words = words.lower()
    tokens = alpha_tokenizer.tokenize(words)
    model_tokens = []
    for index, word in enumerate(tokens):
        if lemmatize:
            lem_word = lemmer.lemmatize(word)
            if lem_word in model_lemmatized.wv.vocab:
                model_tokens.append(lem_word)
        elif stem:
            stem_word = stemmer.stem(word)
            if stem_word in model_stemmed.wv.vocab:
                model_tokens.append(stem_word)
        else:
            if word in model.wv.vocab:
                model_tokens.append(word)
    return model_tokens

old_err_state = np.seterr(all='raise')

def vectorize(words, words_2, model, num_features, lemmatize=False, stem=False):
    features = np.zeros((num_features), dtype='float32')
    words_amount = 0
    
    words = preprocess_check(words, lemmatize, stem)
    words_2 = preprocess_check(words_2, lemmatize, stem)
    for word in words: 
            words_amount = words_amount + 1
            features = np.add(features, model.wv.__getitem__(word))
    for word in words_2: 
            words_amount = words_amount + 1
            features = np.add(features, model.wv.__getitem__(word))
    try:
        features = np.divide(features, words_amount)
    except FloatingPointError:
        features = np.zeros(num_features, dtype='float32')
    return features


#df_test = pandas.read_csv('/Users/carl/Desktop/UWO/Machine Learning/project/test.csv').fillna('None')
q1 = df_train.question1.values[:3000]
q2 = df_train.question2.values[:3000]
q1t = df_train.question1.values[3000:5000]
q2t = df_train.question2.values[3000:5000]
Y1 = np.array(df_train.is_duplicate.values)[:3000]
Yt = np.array(df_train.is_duplicate.values)[3000:5000]

X_train = []
for index, sentence in enumerate(q1):
    X_train.append(vectorize(sentence, q2[index], model, VECTOR_SIZE))
X_train = np.array(X_train)

X_test = []
for index, sentence in enumerate(q1t):
    X_test.append(vectorize(sentence, q2t[index], model, VECTOR_SIZE))
X_test = np.array(X_test)

m1,n1 = np.shape(X_train)
for i in range(m1):
    Y[i] = Y1[i] + 1

import matplotlib.pyplot as plt

#the sigmoid function
def sigmoid(z):
    return np.array(1.0 / (1 + np.exp(-z)), dtype = np.float128)

m1,n1 = np.shape(X_train)
#the cost function
def costfunction(y,h):
    y = np.array(y)
    h = np.array(h)
    y = np.array(y, dtype = np.float128)
    h = np.array(h, dtype = np.float128)
    J1 = sum(y*(np.log(h))+(1-y)*(np.log(1-h)))
    #m = np.shape(x)
    J = - J1 / m1
    return J
# the batch gradient descent algrithm
def gradescent(x,y):
    m,n = np.shape(x)     #m: number of training example; n: number of features
    x = np.c_[np.ones(m),x]     #add x0
    x = np.mat(x)      # to matrix
    y = np.mat(y)
    a = 0.00025       # learning rate
    iteration = 1000
    theta = np.zeros((n+1,1))  #initial theta

    J = []
    for i in range(iteration):
        h = np.array(sigmoid(x*theta),dtype=np.float128)
        #if(i%50 ==0):
        print('h in GD is',h,'\nstep =',i)
        theta = theta + a * (x.T)*(y-h)
        cost = costfunction(y,h)
        print('cost:', cost)
        if cost <=5:
            break
        J.append(cost)

    plt.figure()    
    plt.plot(J)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    print("theta",theta,"cost\n",J)
    return theta,cost
    #print("theta",theta,"cost\n",cost)
    #y_p = classifyVector(X_test,theta)
    #return y_p
    #print(y_p)

#the stochastic gradient descent (m should be large,if you want the result is good)
def stocGraddescent(x,y):
    m,n = np.shape(x)     #m: number of training example; n: number of features
    x = np.c_[np.ones(m),x]     #add x0
    x = np.mat(x)      # to matrix
    y = np.mat(y)
    a = 0.01       # learning rate
    theta = np.ones((n+1,1))    #initial theta

    J = []
    for i in range(m):
        h = np.array(sigmoid(x[i]*theta), dtype = np.float128)
        if(i%50 ==0):
            print('h in GD is %f, step = %f'%(h,i))
        theta = theta + a * x[i].transpose()*(y[i]-h)
        cost = costfunction(y,h)
        print('cost:', cost)
        if cost <=0.5:
            break
        J.append(cost)
    plt.plot(J)
    plt.show()
    return theta,cost

def classifyVector(inX,theta):
    cla = np.array(inX*theta, dtype=np.float128)
    prob = sigmoid((cla).sum(1))
    #print(prob)
    #return prob
    return np.where(prob >= 0.5, 1, 0)

def accuracy(x,y, theta):
    m = np.shape(x)[0]
    c = 0
    x = np.c_[np.ones(m),x]
    y_p = classifyVector(x,theta)
    #return prob
    for i in range(m):
        if y_p[i]==y[i]:
            c = c+1
    accuracy = c/m
    return accuracy

theta,cost=gradescent(X_train,Y1)
print("theta:",theta)
print("J:",cost)
result=accuracy(X_test,Yt,theta)
print("result:",result)

