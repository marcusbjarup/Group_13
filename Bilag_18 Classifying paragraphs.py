# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:59:15 2017

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
#%%
df = df.drop('Unnamed: 0', axis=1)
df = df.drop('international_politik', axis=1)
df = df.drop('miljø_klima', axis=1)
df = df.drop('Kultur_rel_medier', axis=1)
df = df.drop('statsforv_tek', axis=1)
df = df.drop('Konflikter_konsekvenser', axis=1)
df = df.drop('familie_identitet', axis=1)
df = df.drop('andet', axis=1)
df = df.drop('ignore', axis=1)

#%%
import pickle
with open('vec_df[paragraffer].pkl', 'rb') as fin:
    x = pickle.load(fin)
#%%
with open('mod_int_pol.pkl', 'rb') as fin:
    logreg = pickle.load(fin)
    
#%%
prob_int_pol = logreg.predict_proba(x)
#%%
df['int_pol'] = logreg.predict(x)
df['int_pol_p'] = prob_int_pol[:,1]

#%%
with open('mod_fam_id.pkl', 'rb') as fin:
    logreg1 = pickle.load(fin)
    
#%%
fam_id_pred = logreg1.predict(x)
di = {0:1, 1:0}
d = pd.Series(fam_id_pred).map(di)
df['fam_id'] = d
prob_fam_id = logreg1.predict_proba(x)
df['fam_id_p'] = prob_fam_id[:,0]

#%%
with open('mod_igno.pkl', 'rb') as fin:
    logreg2 = pickle.load(fin)
    
#%%
df['ignore'] = logreg2.predict(x)
prob_ignore = logreg2.predict_proba(x)
df['ingore_p'] = prob_ignore[:,1]
#%%
with open('mod_kon_kon.pkl', 'rb') as fin:
    logreg3 = pickle.load(fin)
#%%
df['kon_kon'] = logreg3.predict(x)
prob_kon_kon = logreg3.predict_proba(x)
df['kon_kon_p'] = prob_kon_kon[:,1]
#%%
with open('mod_kul_rel.pkl', 'rb') as fin:
    logreg4 = pickle.load(fin)
#%%
df['kul_rel'] = logreg4.predict(x)
prob_kul_rel = logreg4.predict_proba(x)
df['kul_rel_p'] = prob_kul_rel[:,1]
#%%
with open('mod_mil_kli.pkl', 'rb') as fin:
    logreg5 = pickle.load(fin)
#%%
df['mil_kli'] = logreg5.predict(x)
prob_mil_kli = logreg5.predict_proba(x)
df['mil_kli_p'] = prob_mil_kli[:,1]
#%%
with open('mod_other.pkl', 'rb') as fin:
    logreg6 = pickle.load(fin)
df['other'] = logreg6.predict(x)
prob_other = logreg6.predict_proba(x)
df['other_p'] = prob_other[:,1]
#%%
with open('mod_sta_tek.pkl', 'rb') as fin:
    logreg7 = pickle.load(fin)
sta_tek_pred = logreg7.predict(x)
di = {0:1, 1:0}
d = pd.Series(sta_tek_pred).map(di)
df['sta_tek'] = d
prob_sta_tek = logreg7.predict_proba(x)
df['sta_tek_p'] = prob_sta_tek[:,0]
#%%
with open('pred_val.pkl', 'wb') as fout:
    pickle.dump((df), fout)