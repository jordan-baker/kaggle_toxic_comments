#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:59:35 2018

@author: jordanbaker

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

# load base packages
import os
import numpy as np
import pandas as pd

# load nlp and modeling packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

path = "/Users/jordanbaker/Documents/Data Science/kaggle_toxic"
os.chdir(path)

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# check if any columns contain nulls
train.isnull().sum()
test.isnull().sum()

# calculate class rates
np.sum(train['toxic'][train['toxic']==1])/len(train) # 9.58%
np.sum(train['severe_toxic'][train['severe_toxic']==1])/len(train) # 0.99%
np.sum(train['obscene'][train['obscene']==1])/len(train) # 5.29%
np.sum(train['threat'][train['threat']==1])/len(train) # 0.30%
np.sum(train['insult'][train['insult']==1])/len(train) # 4.94%
np.sum(train['identity_hate'][train['identity_hate']==1])/len(train) # 0.88%

# combine train and test comments together for entire corpus
both = pd.concat([train['comment_text'], test['comment_text']], axis=0)
train_rows = train.shape[0]

# convert a the corpus of comments into a sparse tf-idf matrix
# tf is term frequency: how often the word occurs, with low occurences being deemed 'important'
# idf is inverse document frequency: weights (using log function) applied to account for words being used more/less
vectorizer = TfidfVectorizer(stop_words='english', max_features=100000)
data = vectorizer.fit_transform(both)

# why do we do this?
x = MaxAbsScaler().fit_transform(data)

# define output categories
cats = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# initialize predictions to a matrix of 0s
# initialize loss list
preds = np.zeros((test.shape[0], len(cats)))
loss = []

# for model performance we will use log loss
# log loss estimates model accuracy by penalizing false classifications
# minimizing log loss is basically equivalent to maximizing accuracy
# log Loss heavily penalises classifiers that are confident about an incorrect classification
# http://www.exegetic.biz/blog/2015/12/making-sense-logarithmic-loss/
for i, j in enumerate(cats):
    print('Fit '+j)
    model = LogisticRegression()
    model.fit(x[:train_rows], train[j])
    preds[:,i] = model.predict_proba(x[train_rows:])[:,1]
    
    pred_train = model.predict_proba(x[:train_rows])[:,1]
    print('log loss:', log_loss(train[j], pred_train))
    loss.append(log_loss(train[j], pred_train))
    
print('mean column-wise log loss:', np.mean(loss))

# diminishing returns as max features increases, but perhaps try more
# max_features=10000, mean log loss=0.04704559438912459
# max_features=25000, mean log loss=0.044891590723050594
# max_features=50000, mean log loss=0.04358703284081411
# max_features=100000, mean log loss=0.04224381270661481











# maybe new performance metric?
# this is a repeat of code that starts at line 68
for i, j in enumerate(cats):
    print('Fit '+j)
    model = LogisticRegression()
    model.fit(x[:train_rows], train[j])
    preds[:,i] = model.predict_proba(x[train_rows:])[:,1]
    
    pred_train = model.predict_proba(x[:train_rows])[:,1]
    print('log loss:', log_loss(train[j], pred_train))
    loss.append(log_loss(train[j], pred_train))
    
print('mean column-wise log loss:', np.mean(loss))

# submission code
# MAKE SURE TO UPDATE SUBMISSION NUMBER
# last submission number: 1
sub = pd.DataFrame({'id': sample_sub["id"]})
submission = pd.concat([sub, pd.DataFrame(preds, columns = cats)], axis=1)
submission.to_csv('submission1.csv', index=False)

# https://www.kaggle.com/yekenot/toxic-regression/code

#==============================================================================
# TO DO

# better commenting/explanations
# check out model zoo? or something similar to that?
# jupyter?
# review kaggle discussion boards
# add visualization
# review model block of code
# cross validate results
# try new model approach? check kaggle for this
# alternate data sources? check kaggle for these
# final writeup
# post to github io

#==============================================================================


#==============================================================================
# REFERENCES
# https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras/notebook
# new base model? https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/47964
#==============================================================================












