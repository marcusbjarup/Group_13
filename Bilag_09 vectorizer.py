# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:07:18 2017

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
df = pd.read_csv('df_med paragraffer.csv', encoding='ISO-8859-1')
df = df.drop('Unnamed: 0', axis=1)
#%%
x = vectorizer.transform(df['paragraffer'])

#%%
import pickle 
with open('vec_df[paragraffer].pkl', 'wb') as fout:
    pickle.dump((x), fout)