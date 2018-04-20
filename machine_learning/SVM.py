#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:18:26 2018

@author: carl
"""

import numpy as np
import random

np.set_printoptions(suppress=True)

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import LancasterStemmer
stemmer =  LancasterStemmer()
lemmer = WordNetLemmatizer()
#print(stemmer.stem('dictionaries'))
#print(lemmer.lemmatize('dictionaries'))

from gensim import models
import numpy as np

import pandas

import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
stop = stopwords.words('english')
alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')

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
#model_lemmatized.wv.most_similar('playstation')

#q1 = df_train.question1.values[200:]
#q2 = df_train.question2.values[200:]
#Y = np.array(df_train.is_duplicate.values)[200000:]

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

#X_lem = []
#for index, sentence in enumerate(q1):
    #X_lem.append(vectorize(sentence, q2[index], model_lemmatized, VECTOR_SIZE, True, False))
#X_lem = np.array(X_lem)

#X_stem = []
#for index, sentence in enumerate(q1):
    #X_stem.append(vectorize(sentence, q2[index], model_stemmed, VECTOR_SIZE, False, True))
#X_stem = np.array(X_stem)

#X = []
#for index, sentence in enumerate(q1):
    #X.append(vectorize(sentence, q2[index], model, VECTOR_SIZE))
#X = np.array(X)
#print("1",X)
#print(len(X))

results = []
title_font = {'size':'10', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} 
axis_font = {'size':'10'}

plt.figure(figsize=(10, 5))
plt.xlabel('Training examples', **axis_font)
plt.ylabel('Accuracy',  **axis_font)
plt.tick_params(labelsize=10)

#for X_set, name, lstyle in [(X_lem, 'Lemmatizaton', 'dotted'),
            #(X_stem, 'Stemming', 'dashed'),
            #(X, 'Default', 'dashdot'),
            #]:
    #estimator = LogisticRegression(C = 1)
    #cv = ShuffleSplit(n_splits=6, test_size=0.01, random_state=0)
    #train_sizes=np.linspace(0.01, 0.99, 6)
    #train_sizes, train_scores, test_scores = learning_curve(estimator, X_set, Y, cv=cv, train_sizes=train_sizes)
    #train_scores_mean = np.mean(train_scores, axis=1)
    #results.append({'preprocessing' : name, 'score' : train_scores_mean[-1]})
    #plt.plot(train_sizes, train_scores_mean, label=name, linewidth=5, linestyle=lstyle)
   

plt.legend(loc='best')

#clf = LogisticRegression(C = 1)
#clf.fit(X, Y)

df_test = pandas.read_csv('/Users/carl/Desktop/UWO/Machine Learning/project/train.csv').fillna('None')
q1 = df_train.question1.values[:20000]
q2 = df_train.question2.values[:20000]
q1_test = df_train.question1.values[20000:22000]
q2_test = df_train.question2.values[20000:22000]
Y = np.array(df_train.is_duplicate.values)[:20000]
Y_test = np.array(df_train.is_duplicate.values)[20000:22000]

X_train = []
for index, sentence in enumerate(q1):
    X_train.append(vectorize(sentence, q2[index], model, VECTOR_SIZE))
X_train = np.array(X_train)
print("2:",X_train)
print(len(X_train))

X_test = []
for index, sentence in enumerate(q1_test):
    X_test.append(vectorize(sentence, q2_test[index], model, VECTOR_SIZE))
X_test = np.array(X_test)
print("2:",X_test)
print(len(X_test))
#result = clf.predict(X_test)
#probab = clf.predict_proba(X_test)
#print(probab)
#result = np.gradescent(X_test,Y)
#print("result:",result)
class smoTrain(object):
    def __init__(self, kernel='linear', degree=2, C=0.1, gamma=None, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.tol = tol
        self.degree = degree

    def _kernel(self, x, z=None):
        if z is None:
            z = x
        if self.kernel == 'linear':
            return self.linearKernel(x, z)
        elif self.kernel == 'poly':
            return self.polyKernel(x, z)
        elif self.kernel == 'rbf':  # K(x,z)=exp(− gamma*|x−z|**2)
            return self.GaussianKernel(x, z)
        else:
            print("kernel error")

    def train(self, X, y):
        # X = np.array(X)
        # y = np.array(y)
        self.X = X
        self.y = y
        k = self._kernel(X)
        self.alpha = np.zeros(len(y))
        self. b = 0.0
        c = 0
        while c < 5:
            # print(count)
            alpha_change = 0
            #iteration i and j to get best alpha
            for i in range(len(self.alpha)):
                print(alpha_change)
                # Calculate error for i
                error_i = self.b + np.sum(self.alpha * y * k[i]) - y[i]
                if((error_i * y[i] < - self.tol) and (self.alpha[i] < self.C)) or \
                  ((error_i * y[i] > self.tol) and (self.alpha[i] > 0)):
                    # print('inside')
                    j = random.randint(0, len(self.alpha) - 1)
                    while i == j:
                        j = random.randint(0, len(self.alpha) - 1)
                    # Calculate error for j
                    error_j = self.b + np.sum(self.alpha * y * k[j]) - y[j]
                    ai = self.alpha[i]  # old alpha i
                    aj = self.alpha[j]  # old alpha j

                    #calculate the boundary L and H for alpha j  
                    if y[i] == y[j]:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                    if L == H:
                        # next iteration if no area
                        print("L==H")
                        continue
                    
                    #calculate eta(the similarity of sample i and j)
                    eta = 2 * k[i, j] - k[i, i] - k[j, j]
                    if eta >= 0:
                        print("eta out")
                        continue

                    # update alpha j
                    self.alpha[j] = aj - y[j] * (error_i - error_j) / eta

                    # clip alpha j  
                    if(self.alpha[j] > H):
                        self.alpha[j] = H
                    if(self.alpha[j] < L):
                        self.alpha[j] = L

                    #if alpha j not moving enough, just return
                    if abs(self.alpha[j] - aj) < self.tol:
                        # if change less than tol
                        self.alpha[j] = aj
                        print(" This alpha j not moving enough")
                        continue

                    # update alpha i after optimizing aipha j
                    self.alpha[i] = ai + y[i] * y[j] * (aj - self.alpha[j])

                    #update threshold b
                    b1 = self.b - error_i - y[i] * (self.alpha[i] - ai) * k[i, i] + \
                        y[j] * (self.alpha[j] - aj) * k[i, j]

                    b2 = self.b - error_i - y[i] * (self.alpha[i] - ai) * k[i, j] + \
                        y[j] * (self.alpha[j] - aj) * k[j, j]

                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    alpha_change += 1
                    print("iter:{0}, change:{1}".format(i, alpha_change))


            if alpha_change == 0:
                # print("count + 1")
                c += 1
            else:
                # print("reset count")
                c = 0

        self.w = np.dot(self.alpha * y, self.X)
        # self.alpha = self.alpha[self.alpha > 0]
        # index = np.where(self.alpha>0)[0]
        # self.X = self.X[index]
        # self.y = self.y[index]

    #def predict(self, px):
    def predict(self, x, z=None):
 
        p = np.dot(self.w, x.T) + self.b
        p[p >= 0] = 1
        p[p < 0] = -1
        
        return p
        

    def linearKernel(self, x, z):
        return np.dot(x, z.T)

    def polyKernel(self, x, z):
        return np.power(np.dot(x, z.T + 1.0), 2)

    def GaussianKernel(self, x, z):
        xx = np.sum(x * x, axis=1)
        zz = np.sum(z * z, axis=1)
        res = - 2.0 * np.dot(x, z.T) + \
            xx.reshape(-1, 1) + \
            zz.reshape(1, -1)
        return np.exp(-1 * self.gamma * res)  # 0< gamma < 1


#f = sio.loadmat('f:\\matlab\Hw2-package\spamTrain.mat')
#ff = sio.loadmat('f:\\matlab\Hw2-package\spamTest.mat')
XX = X_train
testx = X_test
#XX = f['X']
#testx = ff['Xtest']
#yy = f['y']
#testy = ff['ytest'].reshape(1, -1)[0].astype(int)
yy = Y
yp = Y_test
# XX = XX[:2000]
yy = yy.reshape(1, -1)[0].astype(int)
yy[yy == 0] = -1
yp = yp.reshape(1,-1)[0].astype(int)
yp[yp == 0] = -1

#testy[testy == 0] = -1
# print(yy)
#print(testy)

smo = smoTrain('linear')
smo.train(XX, yy)
p = smo.predict(testx)
print("p",p)
count = 0
p = np.array(p)
for i in range(len(p)):
    if p[i] == yp[i]:
        count+=1

print('accuracy:', count/len(yp))
