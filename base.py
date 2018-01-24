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

# load nlp packages
import string
import re    #for regex
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary

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
