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
import string
import nltk
import spacy
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer  
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD

# set working directory
path = "/Users/jordanbaker/Documents/Data Science/kaggle_toxic"
os.chdir(path)

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

eng_stopwords = set(stopwords.words('english'))
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()

# apostrophe dictionary
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

def cleaner(comment):

    comment = comment.lower()
    comment = re.sub('\\n', '', comment)
    comment = re.sub('\[\[.*\]', '', comment)
    
    words = tokenizer.tokenize(comment)
    
    words = [APPO[i] if i in APPO else i for i in words]
    words = [lem.lemmatize(i, 'v') for i in words]
    words = [i for i in words if not i in eng_stopwords]
    
    cleaned = ' '.join(words)

    return(cleaned)

clean_train = train['comment_text'].apply(lambda x: cleaner(x))




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








