import pandas as pd
#%%
j = 1
base = '/home/polichinel/Dropbox/KU/7.semester/SDS/python_n_data/sds_eks/'
endings = []

while j < 382:
    endings.append('text' + str(j) + '.csv')
    j += 1    
#%%
list_all = []
j=1

for i in endings:
    list_all.append(pd.read_csv(base+i))
    
data_all = pd.concat(list_all, ignore_index=True)
len(data_all)