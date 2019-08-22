# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:12:38 2017

@author: The Apocalypse Four
"""

import os
import pandas as pd

os.chdir('C:/Users/Niels/OneDrive - Københavns Universitet/SDS/EXAM')

#%% Importer og sorter data fra kodet paragraffer
df_n = pd.read_excel('Niels_1_sorted.xlsx')
df_e = pd.read_excel('Elias_3_done.xlsx')
df_j = pd.read_excel('Julius_4_Done.xlsx')
df_s = pd.read_excel('Simon_2.4.xlsx')

#%% Lav indeks i hver af de fire dataframes

df_n['article_id'] = df_n.index
df_n['sum_n'] = [df_n.iloc[i,10:18].sum() for i in range(df_n.shape[0])]
df_e['article_id'] = df_e.index
df_e['sum_e'] = [df_e.iloc[i,10:18].sum() for i in range(df_e.shape[0])]
df_j['article_id'] = df_j.index
df_j['sum_j'] = [df_j.iloc[i,10:18].sum() for i in range(df_j.shape[0])]
df_s['article_id'] = df_s.index
df_s['sum_s'] = [df_s.iloc[i,10:18].sum() for i in range(df_s.shape[0])]

#%% Merge data_frames

df_merge1 = pd.merge(pd.merge(pd.merge(df_n, df_e, how = 'outer'), df_j, how = 'outer'),df_s, how='outer' )

df_sum = df_merge1[['sum_n', 'sum_e', 'sum_j','sum_s']]
counts = []
for i in range(df_sum.shape[0]):
    counts.append(sum(df_sum.iloc[i, :].notnull()) > 1)

df_merge1['counts'] = counts
df_overlap = df_merge1.iloc[counts, :]
df_merge1 = df_merge1.fillna(0)

#%% Læg det ned som csv
#df_overlap.to_csv('DF_overlap.csv', encoding='utf-8')
#df_merge1.to_csv('DF_merge.csv', encoding='utf-8')


#%% Væk med nul-rækker
dfXy = pd.DataFrame()
dfXy['int_pol'] = df_merge1[['n_international_politik','international_pol','j_international_politik','s_international_politik']].max(axis=1)
dfXy['mil_kli'] = df_merge1[['n_miljø_klima', 'miljø_klima', 'j_miljø_klima','s_miljø_klima']].max(axis=1)
dfXy['kul_rel'] = df_merge1[['n_Kultur_rel_medier', 'Kultur_rel_medier', 'j_Kultur_rel_medier', 's_Kultur_rel_medier']].max(axis=1)
dfXy['sta_tek'] = df_merge1[['n_statsforv_tek', 'statsforv_tek', 'j_statsforv_tek', 's_statsforv_tek']].max(axis=1)
dfXy['kon_kon'] = df_merge1[['n_Konflikter_konsekvenser', 'Konflikter_konse', 'j_Konflikter_konsekvenser', 's_Konflikter_konsekvenser']].max(axis=1)
dfXy['fam_id']  = df_merge1[['n_familie_identitet', 'familie_identitet', 'j_familie_identitet', 's_familie_identitet']].max(axis=1)
dfXy['other'] = df_merge1[['n_familie_identitet', 'familie_identitet', 'j_familie_identitet', 's_familie_identitet']].max(axis=1)
dfXy['other'] = df_merge1[['n_andet', 'andet', 'j_andet', 's_andet']].max(axis=1)
dfXy['igno'] = df_merge1[['n_ignore','ignore', 'j_ignore', 's_ignore']].max(axis=1) 
dfXy['par_txt'] = df_merge1['paragraffer']
dfXy['article_id'] = df_merge1['article_id']
dfXy['gender'] = df_merge1['gender']

dfXy.to_csv('TestAndTrainMatrix_inclOtherIgnore.csv', encoding='utf-8')


