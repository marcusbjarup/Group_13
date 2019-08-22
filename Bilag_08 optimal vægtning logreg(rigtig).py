#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 19:38:59 2017

@author: polichinel
"""
#%%
import nltk
import sklearn
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import re

#%%
df = pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/Eks/nyt/TestAndTrainMatrix_inclOtherIgnore.csv')
df = df.drop('Unnamed: 0', axis = 1)

#%%

df.iloc[3888, df.columns.get_loc('int_pol')] = 1
df.iloc[3518, df.columns.get_loc('int_pol')] = 1


#%%

print('int_pol:',len(df['int_pol'][df['int_pol'] == 1]))
print('mil_kli',len(df['mil_kli'][df['mil_kli'] == 1])) #kun 159..
print('kul_rel',len(df['kul_rel'][df['kul_rel'] == 1]))
print('fam_id',len(df['fam_id'][df['fam_id'] == 1]))
print('sta_tek',len(df['sta_tek'][df['sta_tek'] == 1]))
print('other',len(df['other'][df['other'] == 1]))
print('igno',len(df['igno'][df['igno'] == 1]))



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
#+++++++++++++++++++++++++ Weights and model Int pol log reg ++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['int_pol']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:3.0204081632653064} 

logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model fam_id log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['fam_id']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:5.591836734693878} 
#

logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model kon_kon log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['kon_kon']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:4.3061224489795915} 
#

logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model kul_rel reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['kul_rel']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:6.1428571428571432} 
#
logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model mil-kli reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['mil_kli']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:13.408163265306122} 
#class_weight= cl_weight, 
logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 20
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model other log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['other']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:10.306122448979592} 
#
logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 20
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model sta_tek log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['sta_tek']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:2.1020408163265305} 
#class_weight= cl_weight,
logrg = LogisticRegression( solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#%%

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++ Weights and model Igno log reg ++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



X = df['par_txt']
y = df['igno']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

cl_weight = {0:1, 1:2.8367346938775508} 
#
logrg = LogisticRegression(class_weight= cl_weight, solver = 'newton-cg')

logrg.fit(X_train_dtm, y_train)

y_pred_class = logrg.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%% Loop over vÃ¦gte
start = 1
stop = 10
steps = 50
meassures = {'acc_score':[], 'prec_score':[], 'recall_score':[], 'f1_score':[], 'conf_matr':[],'acc_score_o':[], 'prec_score_o':[], 'recall_score_o':[], 'f1_score_o':[], 'weight':[]}
for i in np.linspace(start, stop, steps):
    cl_weight = {0:1, 1:i}
    logreg = LogisticRegression(class_weight=cl_weight,solver='newton-cg')
    logreg.fit(X_train_dtm, y_train)
    y_pred_class = logreg.predict(X_test_dtm)
    acc_score = metrics.accuracy_score(y_test, y_pred_class)
    prec_score = metrics.precision_score(y_test, y_pred_class)
    recall_score = metrics.recall_score(y_test, y_pred_class)
    f1_score =     metrics.f1_score(y_test, y_pred_class)
    confusion = list(metrics.confusion_matrix(y_test, y_pred_class))
    meassures['conf_matr'].append([confusion])
    meassures['acc_score'].append([i, acc_score])
    meassures['prec_score'].append([i, prec_score])
    meassures['recall_score'].append([i, recall_score])
    meassures['f1_score'].append([i, f1_score])
    meassures['acc_score_o'].append([acc_score]) # til plt
    meassures['prec_score_o'].append([prec_score])
    meassures['recall_score_o'].append([recall_score]) # til plt
    meassures['f1_score_o'].append([f1_score]) # til plt
    meassures['weight'].append([i]) # til plt


#%%

plt.plot(meassures['weight'],meassures['acc_score_o'])


#%%

plt.plot(meassures['weight'],meassures['prec_score_o'])



#%%
plt.plot(meassures['weight'],meassures['recall_score_o'])


#%%
plt.plot(meassures['weight'],meassures['f1_score_o'])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++