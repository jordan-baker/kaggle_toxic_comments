#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:59:35 2018

@author: jordanbaker

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
"""
# load base packages
import os
import pandas as pd
import numpy as np
import random
random.seed(312)

# nlp
import re

# set working directory
path = "/Users/jordanbaker/Documents/Data Science/kaggle_toxic"
os.chdir(path)

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# calculate class rates
np.sum(train['toxic'][train['toxic']==1])/len(train) # 9.64%
np.sum(train['severe_toxic'][train['severe_toxic']==1])/len(train) # 1.01%
np.sum(train['obscene'][train['obscene']==1])/len(train) # 5.33%
np.sum(train['threat'][train['threat']==1])/len(train) # 0.32%
np.sum(train['insult'][train['insult']==1])/len(train) # 4.97%
np.sum(train['identity_hate'][train['identity_hate']==1])/len(train) # 0.85%

# features for special characters
train['!'] = train['comment_text'].apply(lambda x: x.count('!'))
train['?'] = train['comment_text'].apply(lambda x: x.count('?'))
train['@'] = train['comment_text'].apply(lambda x: x.count('@'))
train['#'] = train['comment_text'].apply(lambda x: x.count('#'))
train['$'] = train['comment_text'].apply(lambda x: x.count('$'))
train[','] = train['comment_text'].apply(lambda x: x.count(','))
train['.'] = train['comment_text'].apply(lambda x: x.count('.'))

# features for counts
train['word_count'] = train['comment_text'].apply(lambda x: len(x.split()))
train['unique_word_count'] = train['comment_text'].apply(lambda x: len(set(x.split())))
train['sent_count'] = train['comment_text'].apply(lambda x: len(re.findall("\n", str(x)))+1)
train['punct_count'] = train['comment_text'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]))
train['upper_count'] = train['comment_text'].apply(lambda x: len([i for i in str(x).split() if i.isupper()]))
train['title_count'] = train['comment_text'].apply(lambda x: len([i for i in str(x).split() if i.istitle()]))
train['stopword_count'] = train['comment_text'].apply(lambda x: len([i for i in str(x).lower().split() if i in eng_stopwords]))







