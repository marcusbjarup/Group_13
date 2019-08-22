# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 17:06:03 2017

@author: Julius Markedal
"""

############################################################################
############ Confusion Matrix DataFrame ####################################
############################################################################
import pandas as pd
import re
import nltk
import sklearn
from sklearn import metrics
from sklearn.cross_validation import train_test_split as tts
from sklearn.linear_model import LogisticRegression
#%%
df_matrix = pd.DataFrame([])
#%%
df= pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/Eks/nyt/TestAndTrainMatrix_inclOtherIgnore.csv')
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
Emne = ['int_pol', 'fam_id', 'ignorer', 'kon_kon', 'kul_rel', 'mil_kli', 'other', 'sta_tek']
true_positive = []
false_positive = []
true_negative = []
false_negative = []
f1 = []
accuracy = []
recall = []
precision = []

#%%
# Int_pol---------------------------------------------------------------------
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

print(metrics.confusion_matrix(y_test, y_pred_class))

#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0

#%%
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# Fam_id----------------------------------------------------------------------

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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
sum(bool & bool4)
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool3 & bool4))
false_positive.append(sum(bool2 & bool3))
true_negative.append(sum(bool1 & bool2))
false_negative.append(sum(bool4 & bool1))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# Ignorer

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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# kon_kon----------------------------------------------------------------------
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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# Kul_rel---------------------------------------------------------------------

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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# mil_kli----------------------------------------------------------------------

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


print(metrics.confusion_matrix(y_test, y_pred_class))

#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# other-----------------------------------------------------------------------

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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool1 & bool2))
false_positive.append(sum(bool1 & bool4))
true_negative.append(sum(bool3 & bool4))
false_negative.append(sum(bool3 & bool2))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%
# Sta_tek----------------------------------------------------------------------

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

print(metrics.confusion_matrix(y_test, y_pred_class))
#%%
bool1 = y_pred_class == 1
bool2 = y_test == 1
bool3 = y_pred_class == 0
bool4 = y_test == 0
true_positive.append(sum(bool3 & bool4))
false_positive.append(sum(bool2 & bool3))
true_negative.append(sum(bool1 & bool2))
false_negative.append(sum(bool4 & bool1))
accuracy.append(metrics.accuracy_score(y_test, y_pred_class))
recall.append(metrics.recall_score(y_test, y_pred_class))
precision.append(metrics.precision_score(y_test, y_pred_class))
f1.append(metrics.f1_score(y_test, y_pred_class))
#%%

df_matrix['emne'] = Emne
df_matrix['true_positive'] = true_positive
df_matrix['false_positive'] = false_positive
df_matrix['true_negative'] = true_negative
df_matrix['false_negative'] = false_negative
df_matrix['accuracy'] = accuracy
df_matrix['recall'] = recall
df_matrix['precision'] = precision
df_matrix ['f1'] = f1
#%%
df_matrix.set_index('emne', inplace=True)

#%%
writer = pd.ExcelWriter('ConfusionMatrixTable1.xlsx')
df_matrix.to_excel(writer,'Sheet1')
writer.save()