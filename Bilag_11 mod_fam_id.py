# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:53:08 2017

@author: Julius Markedal
"""

#%%
import nltk
import sklearn
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
import re
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt

#%%
df = pd.read_csv('C:/Users/Julius Markedal/desktop/python/sds/TestAndTrainMatrix_inclOtherIgnore.csv')
df = df.drop('Unnamed: 0', axis = 1)


#%%
stemmer = nltk.stem.snowball.DanishStemmer()
def prep(text):
    wordlist = nltk.word_tokenize(text)
    wordlist = [stemmer.stem (w) for w in wordlist]
    pattern = '^[,;:?Â«<>Â»]+$'
    text = re.sub(pattern,'', text)
    
    return wordlist

def custom_tokenize(text):
   
    text = re.sub('^[,;:?Â«<>Â»]+$','',text)
    
    wordlist = prep(text) # our old function
    wordlist = [word.strip('.,"?') for word in wordlist]
    return wordlist

#%%
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model fam_id log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) # Tfidf
X  = vectorizer.fit_transform(df['par_txt'])
y,y_index = pd.factorize(df['fam_id'])

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

cl_weight = {0:1, 1:0.25510204081632654} 

logreg1 = LogisticRegression(class_weight=cl_weight,solver='newton-cg')

# train the model using X_train_dtm
logreg1.fit(X_train, y_train)

# make class predictions for X_test_dtm
y_pred_class = logreg1.predict(X_test)

# calculate predicted probabilities for X_test_dtm (well calibrated)
#y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]

print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))

#print('roc_curve: ',metrics.roc_curve(y_test, y_pred_class))
#print('roc_auc_curve: ',metrics.roc_auc_score(y_test, y_pred_class))
#%%
import pickle
with open('mod_fam_id.pkl', 'wb') as fout:
    pickle.dump((logreg1), fout)