#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 22:18:54 2018

@author: jordanbaker
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""

# load base packages
import os
import numpy as np
import pandas as pd

# load nlp/modeling packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_score

# set path
path = "/Users/jordanbaker/Documents/Data Science/kaggle_toxic"
os.chdir(path)

# load training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# check if any columns contain nulls
train.isnull().sum()
test.isnull().sum()

# establish prediction classes
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# calculate class rates
np.sum(train['toxic'][train['toxic']==1])/len(train) # 9.58%
np.sum(train['severe_toxic'][train['severe_toxic']==1])/len(train) # 0.99%
np.sum(train['obscene'][train['obscene']==1])/len(train) # 5.29%
np.sum(train['threat'][train['threat']==1])/len(train) # 0.30%
np.sum(train['insult'][train['insult']==1])/len(train) # 4.94%
np.sum(train['identity_hate'][train['identity_hate']==1])/len(train) # 0.88%

# create training text, testing text, and combined text
train_text = train['comment_text']
test_text = test['comment_text']
comb_text = pd.concat([train_text, test_text])

# convert a the corpus of comments into a sparse tf-idf matrix
# 
# tf is term frequency: how often the word occurs, with low occurences
# being deemed 'important'

# idf is inverse document frequency: weights (using log function) applied to
# account for words being used more/less
word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', 
                                  analyzer='word', token_pattern=r'\w{1,}', 
                                  stop_words='english', ngram_range=(1, 1),
                                  max_features=100000)

# fit the vectorizer to all text (so that all words are observed)
# generate testing and training features using the fitted vectorizer
word_vectorizer.fit(comb_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

# same concept as above, but at the character level
char_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', 
                                  analyzer='char', stop_words='english', 
                                  ngram_range=(1, 6), max_features=100000)

# fit the vectorizer to all text (so that all ngrams are observed)
# generate testing and training features using the fitted vectorizer
char_vectorizer.fit(comb_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

# generate training and testing features using word and char features
train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

# empty scores list and predictions dataframe
scores = []
pred = pd.DataFrame.from_dict({'id': test['id']})

# loop through each class, train the ridge model, and make predictions
for class_name in classes:
    train_target = train[class_name]
    classifier = Ridge(alpha=20, fit_intercept=True, solver='auto',
                       max_iter=100, random_state=0, tol=0.0025)
    classifier.fit(train_features, train_target)
    pred[class_name] = classifier.predict(test_features)

# output predictions to csv file
pred.to_csv('predictions7.csv', index=False)

# 50k words, 50k chars, score = 0.9811, file = predictions
# 50k words, 10k chars, score = 0.9805, file = predictions2
# 10k words, 50k chars, score = 0.9809, file = predictions3
# 50k words, 100k chars, score = 0.9812, file = predictions4
# 100k words, 100k chars, score = 0.9812, file = predictions5
# same words/chars, ngram range is (2,5), score = 0.9811, file = predictions6
# same words/chars, ngram range is (1,6), score = 0.9811, file = predictions7

# final to do
# pep8 commenting
# final writeup
# blog post?









