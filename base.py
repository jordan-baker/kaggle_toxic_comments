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
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
data = vectorizer.fit_transform(both)


