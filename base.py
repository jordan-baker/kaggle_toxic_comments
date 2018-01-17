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

from sklearn.linear_model import LogisticRegression

from scipy.sparse import csr_matrix, hstack

# set working directory
path = "/Users/jordanbaker/Documents/Data Science/kaggle_toxic"
os.chdir(path)

# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# check and replace nulls with unknown
train.isnull().sum()
test.isnull().sum()
train['comment_text'].fillna('unknown', inplace=True)
test['comment_text'].fillna('unknown', inplace=True)

# calculate class rates
np.sum(train['toxic'][train['toxic']==1])/len(train) # 9.64%
np.sum(train['severe_toxic'][train['severe_toxic']==1])/len(train) # 1.01%
np.sum(train['obscene'][train['obscene']==1])/len(train) # 5.33%
np.sum(train['threat'][train['threat']==1])/len(train) # 0.32%
np.sum(train['insult'][train['insult']==1])/len(train) # 4.97%
np.sum(train['identity_hate'][train['identity_hate']==1])/len(train) # 0.85%

# merge comments into one corpus so that all vocab is included
both = pd.concat([train.iloc[:,0:2], test.iloc[:,0:2]])
both = both.reset_index(drop=True)

# set stopwords
eng_stopwords = set(stopwords.words('english'))

# features for special characters
both['!'] = both['comment_text'].apply(lambda x: x.count('!'))
both['?'] = both['comment_text'].apply(lambda x: x.count('?'))
both['@'] = both['comment_text'].apply(lambda x: x.count('@'))
both['#'] = both['comment_text'].apply(lambda x: x.count('#'))
both['$'] = both['comment_text'].apply(lambda x: x.count('$'))
both[','] = both['comment_text'].apply(lambda x: x.count(','))
both['.'] = both['comment_text'].apply(lambda x: x.count('.'))

# features for counts
both['word_count'] = both['comment_text'].apply(lambda x: len(x.split()))
both['unique_word_count'] = both['comment_text'].apply(lambda x: len(set(x.split())))
both['sent_count'] = both['comment_text'].apply(lambda x: len(re.findall("\n", str(x)))+1)
both['punct_count'] = both['comment_text'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]))
both['upper_count'] = both['comment_text'].apply(lambda x: len([i for i in str(x).split() if i.isupper()]))
both['title_count'] = both['comment_text'].apply(lambda x: len([i for i in str(x).split() if i.istitle()]))
both['stopword_count'] = both['comment_text'].apply(lambda x: len([i for i in str(x).lower().split() if i in eng_stopwords]))

target_cols = train.columns[2:]
extra_cols = both.columns[2:]

train_extra = both.iloc[0:len(train),][extra_cols]
test_extra = both.iloc[len(train):,][extra_cols]

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

# set up tokenizer and lemmatizer
tokenizer = TweetTokenizer()
lem = WordNetLemmatizer()

# create cleaning function
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

# apply cleaning function to comments
corpus = both['comment_text']
clean_corpus = corpus.apply(lambda x: cleaner(x))

# initialize tf-idf vectorizer
tfv = TfidfVectorizer(min_df=100, max_df=0.9, max_features=100000,
                      strip_accents='unicode', analyzer='word',
                      ngram_range=(1,2), use_idf=1,
                      smooth_idf=1, sublinear_tf=1, stop_words = 'english')

# fit tfv to the cleaned corpus
# translate features to an array
tfv.fit(clean_corpus)
features = np.array(tfv.get_feature_names())

# create bigrams for training and testing sets
train_bigrams =  tfv.transform(train.comment_text.astype('U'))
test_bigrams = tfv.transform(test.comment_text.astype('U'))

train_x = hstack((train_bigrams, train_extra)).tocsr()
test_x = hstack((test_bigrams, test_extra)).tocsr()

def pr(y_i, y):
    p = train_x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)
    
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = train_x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(target_cols)))

for i, j in enumerate(target_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

submid = pd.DataFrame({'id': sample_sub['id']})
submission = pd.concat([submid, pd.DataFrame(preds, columns=target_cols)], axis=1)
submission.to_csv('submission_save.csv', index=False)


# add new features
# try out new ngrams
# try out entirely new modeling approach
# grid search parameters
# add visualization

#ngram testing
#ngram_range=(1,2)
#ngram_range=(1,2)
#ngram_range=(1,2)
#ngram_range=(1,2)
#ngram_range=(1,2)
#ngram_range=(1,2)





