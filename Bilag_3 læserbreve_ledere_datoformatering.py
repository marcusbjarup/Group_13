import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

#%%
def leder_laeserbreve(dataframe):
    """
    Funktioner der opretter leder og læserbrevevariabel
    Input: DataFrame
    Ouput: DataFrame med to yderligere variable
    """
    dataframe['leder'] = [1 if '/debat/leder/' in li else 0 for li in df['links']]
    dataframe['lbreve'] = [1 if '/laeserbreve-' in li else 0 for li in df['links']]
    return dataframe

#%%
#datoer = df.loc[:, 'date']
#datoer.sample(20)
mon = {'januar':'Jan',
       'februar':'Feb',
       'marts':'Mar', 
       'april':'Apr',
       'maj':'May',
       'juni':'Jun',
       'juli':'Jul',
       'august':'Aug',
       'september':'Sep',
       'oktober':'Oct',
       'november':'Nov',
       'december':'Dec'}

#%%
def dato_omskriv(ds):
    try:
        ds = str(ds)
        #print(ds)
        dag = ds.replace('.','')
        #print(dag)
        mdr = dag.split(' ')[1]
        dag = dag.replace(mdr,mon[mdr])
        dat = datetime.strptime(dag, '%d %b %Y')
    except (IndexError, TypeError, UnboundLocalError, KeyError):
        dat = np.nan
    return dat