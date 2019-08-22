# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:58:35 2017

@author: polichinel
"""

#%%
import nltk
import sklearn
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np
import pandas as pd
from sklearn import metrics

#%%
df = pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/Eks/nyt/TestAndTrainMatrix_inclOtherIgnore.csv')
df = df.drop('Unnamed: 0', axis = 1)
df.iloc[3888, df.columns.get_loc('int_pol')] = 1
df.iloc[3518, df.columns.get_loc('int_pol')] = 1

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

df.keys()    
#%%
###############################################################################
########################## Optimerede par. int_pol ############################
###############################################################################

X = df['par_txt']
y = df['int_pol']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]} 

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['int_pol']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) 
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.14371428571428571, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. mil_kli ############################
###############################################################################

X = df['par_txt']
y = df['mil_kli']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  
# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['mil_kli']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.041775510204081635, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################


#%%
###############################################################################
########################## Optimerede par. fam_id ############################
###############################################################################

X = df['par_txt']
y = df['fam_id']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  
# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['fam_id']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.021387755102040818, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. kon_kon ############################
###############################################################################

X = df['par_txt']
y = df['kon_kon']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['kon_kon']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) 

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.062163265306122456, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. kul_rel ############################
###############################################################################

X = df['par_txt']
y = df['kul_rel']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]}  #, 'class_weight': weights

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['kul_rel']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.24565306122448982, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. sta_tek ############################
###############################################################################

X = df['par_txt']
y = df['sta_tek']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]} 

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['sta_tek']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.40875510204081633, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. other ############################
###############################################################################

X = df['par_txt']
y = df['other']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]} 

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['int_pol']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(alpha=0.6126326530612245, class_prior=[0.1, 0.9],fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################

#%%
###############################################################################
########################## Optimerede par. igno ############################
###############################################################################

X = df['par_txt']
y = df['igno']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize)

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


#%%

# Setup the hyperparameter grid
alpha = np.linspace(0.001, 1)
param_grid = {'alpha': alpha, 'fit_prior' : [True, False], 'class_prior' : [None, [.1,.9],[.2, .8]]} 

# Instantiate the GridSearchCV
nb_cv = GridSearchCV(nb, param_grid, cv=5, scoring='f1', n_jobs= -1)

# Fit it to the data
nb_cv.fit(X_train_dtm, y_train)


#%%
print(nb_cv.best_score_)
print(nb_cv.best_params_)
print(nb_cv.best_estimator_)
#%%
# Nu med optimerede parametre!

X = df['par_txt']
y = df['igno']

X_train, X_test, y_train, y_test = tts(X, y, random_state=1)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=custom_tokenize) # Tfidf

X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

nb = MultinomialNB(0.24565306122448982, class_prior=None, fit_prior=True)

nb.fit(X_train_dtm, y_train)

y_pred_class = nb.predict(X_test_dtm)


# Score
print(metrics.confusion_matrix(y_test, y_pred_class))
print('acc: ',metrics.accuracy_score(y_test, y_pred_class))
print('recall: ',metrics.recall_score(y_test, y_pred_class))
print('prec: ',metrics.precision_score(y_test, y_pred_class))
print('f1: ',metrics.f1_score(y_test, y_pred_class))


###############################################################################
###############################################################################
###############################################################################