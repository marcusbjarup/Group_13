import pandas as pd
import numpy as np
import sklearn
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split as tts
import statsmodels.formula.api as smf
import nltk
sns.set
%matplotlib inline

#%%
df1 = pd.read_csv('prepro_v3.csv', encoding='ISO-8859-1') ## Hent DataFrame

#%%
def text_to_paragraphs(tekster):
    """
    Funktionens formål er at dele alle tekster i vores dataframe i deres paragraffer, og derefter folde dataframen ud,
    således at alle paragraffer har en row for sig selv.
    Input: en list med strings, hvor vores strings er lig med vores tekster.
    Output: en dataframe.
    """
    paragraffs = [] # holdinglist til alle paragraffer (resultatet bliver en liste i en liste)
    ### looper over alle tekster.
    for i in tekster:
        try:
            val = nltk.tokenize.line_tokenize(i) #tokenizer på linjeskift(paragraffer) 
            paragraffs.append(val) 
        except (AttributeError) as err: ## printer Error, som value, når der ikke forekommer en tekst
            val = 'Error'               ## Vi printer Error, som value, for at beholde det rette index
            paragraffs.append(val)      ##

    df1['paragraffs'] = paragraffs # paragrafferne tilføjes til DF, som en liste med strings.
    # der laves en ny dataframe, hvor listerne paragraffer foldes ud.
    df_para = df1.apply(lambda x: pd.Series(x['paragraffs']), axis=1).stack().reset_index(level=1, drop=True)
    df_para.name = 'paragraffer'
    # dataframes merges til en endelig datafram, hvor alle paragraffer er foldet ud. 
    df1_p = df1.drop('paragraffs', axis=1).join(df_para)
    df1_p = df1_p.drop('Unnamed: 0', axis=1)
    # konverter alt text til lower case
    df1_p = df1_p.apply(lambda x: x.astype(str).str.lower())
    emner = ['international_politik','miljø_klima','Kultur_rel_medier','statsforv_tek','Konflikter_konsekvenser','familie_identitet','andet','ignore']
    for i in emner:
        df1_p[i] = pd.Series()
    return df1_p

#%%
df1_p = text_to_paragraphs(df1['text'])
df1_p.to_csv('df_med paragraffer.csv')

#%%
def random_sample(sample_size):
    df1_p_sample = df1_p.sample(n=sample_size, random_state=1)
    """
    Funktionens formål er at lave et tilfældigt udtræk af paragraffer fra vores dataframe, og fordele dem tilfældigt i
    4 samples. dernæst lægges 
    Input: en list med strings, hvor vores strings er lig med vores tekster.
    Output: en dataframe.
    """
    # generate 2000 random rows for sampling
    j = int(sample_size/4)
    # split the samples into 4
    part1_sample1 = df1_p_sample[0:j-1]
    part1_sample2 = df1_p_sample[j:j*2-1]
    part1_sample3 = df1_p_sample[j*2:j*3-1]
    part1_sample4 = df1_p_sample[j*3:sample_size-1]
     # concatenate all combinations of 3 parts of the sample
    f1 = pd.concat([pd.concat([part1_sample2, part1_sample3]), part1_sample4])
    f2 = pd.concat([pd.concat([part1_sample1, part1_sample3]), part1_sample4])
    f3 = pd.concat([pd.concat([part1_sample1, part1_sample2]), part1_sample4])
    f4 = pd.concat([pd.concat([part1_sample1, part1_sample2]), part1_sample3])
    # generate a random sample of the groups of 3
    part2_sample1 = f1.sample(frac=0.0333, random_state=2)
    part2_sample2 = f2.sample(frac=0.0333, random_state=3)
    part2_sample3 = f3.sample(frac=0.0333, random_state=4)
    part2_sample4 = f4.sample(frac=0.0333, random_state=5)
    # concat every part of the sample, with the randomly generated sample from the group, that it was not a part of.
    sample1 = pd.concat([part1_sample1, part2_sample1])
    sample2 = pd.concat([part1_sample2, part2_sample2])
    sample3 = pd.concat([part1_sample3, part2_sample3])
    sample4 = pd.concat([part1_sample4, part2_sample4])
    return sample1, sample2, sample3, sample4

#%%
samples = random_sample(4000)

#%%
from pandas import ExcelWriter
# gem samples som excel filer
writer1 = ExcelWriter('Niels_1.xlsx')
samples[0].to_excel(writer1,'Sheet1')
writer1.save()

writer2 = ExcelWriter('Simon_2.xlsx')
samples[1].to_excel(writer2,'Sheet1')
writer2.save()

writer3 = ExcelWriter('Elias_3.xlsx')
samples[2].to_excel(writer3,'Sheet1')
writer3.save()

writer4 = ExcelWriter('Julius_4.xlsx')
samples[3].to_excel(writer4,'Sheet1')
writer4.save()