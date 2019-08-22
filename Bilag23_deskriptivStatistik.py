path = 'C:/Users/Niels/OneDrive - Københavns Universitet/SDS/EXAM/pred_val/pred_val/'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import os
import pickle

#sns.set(style="white")
plt.style.use('seaborn-whitegrid')
os.chdir(path)

ubLil = '#772953'
ubOra = '#E95420'
ubfont = {'fontname':'Calibri'}


#%% 
with open('pred_val.pkl', 'rb') as fin:
    df = pickle.load(fin)

#%% Lav aar variabel, køn
df['yr'] = [int(df['date_time'][i].split("tm_")[1].strip(" ,").split("=")[1]) for i in range(df.shape[0])]
di = {'male': 'all male', 'female': 'all female'}
df = df.replace({'gender':di})
del di
df = df.drop(df[df['ignore'] == 1].index)
#df.gender.value_counts()
df_mf = df.drop(df[df['gender']=='mixed'].index)

#%% udvikling af paragraffer over år
####################################################
### Figuren er klar til dokumentet #################
####################################################
tab_yr = df_mf['yr'].value_counts()
tab_yr = tab_yr.drop(11, axis=0)
tab_yr = tab_yr.sort_index(axis=0)
years = np.arange(1998, 2017)

fig, ax = plt.subplots()
ax.grid(b=None)
width = 0.8
N = len(years)
bar1 = ax.bar(np.arange(N), tab_yr, width, color='#772953', edgecolor = "none")
ax.set_xticks(np.arange(N)+ width/2)
ax.set_frame_on(b=False)
ax.set_xticklabels(years,**ubfont, fontweight='bold')
ax.set_xlim([-0.2, 19])
plt.show()


#%% Plot over køn inkl mixed - ikke klar
tab_gen1 = df['gender'].value_counts()/len(df)
tab_gen1.plot(kind='bar', color = 'orange')


#%% Mænd og kvinder # Kommer ikke med

####################################################
### Figuren er klar til dokumentet #################
####################################################
tab_gen = df_mf['gender'].value_counts()/len(df_mf)

"{:.0%}".format(tab_gen[0])

ubfont = {'fontname':'Calibri'}
fig, ax = plt.subplots()
ax.grid(b=None)
x = tuple(tab_gen)
x1 = "{:.0%}".format(tab_gen[1])
x0 = "{:.0%}".format(tab_gen[0])
width = 0.2
N = 2
ind = np.array([0,0.4])
bar1 = ax.bar(ind[0], x[0], width, color='#772953', edgecolor = "none")
bar2 = ax.bar(ind[1], x[1], width, color='#E95420', edgecolor = "none")
ax.set_xticks(ind + width/2)
ax.set_frame_on(b=False)
ax.set_xticklabels(('Mænd', 'Kvinder'),**ubfont, fontweight='bold')
ax.set_yticklabels([])
ax.set_xlim([-0.2, 0.8])
ax.set_ylim([0, .9])
ax.annotate(x0, (ind[0]+width/2,x[0]-0.05),fontweight='bold',color='white',horizontalalignment='center')
ax.annotate(x1, (ind[1]+width/2,x[1]-0.05),fontweight='bold',color='white',horizontalalignment='center')
plt.show()

#%% Mænd og kvinder - SAMME SOM FØR NU MED 2000 og 2015

####################################################
### Figuren er klar til dokumentet #################
####################################################

keep1 = df_mf['yr']==2000
keep2 = df_mf['yr']==2015
keep3 = keep1 | keep2
df_twoYears = df_mf.loc[keep3, :]
#fem_bool = df_twoYears.gender == 'all female'
#df2y_fem = df_twoYears.loc[fem_bool, :]
emner = ['int_pol', 'sta_tek', 'fam_id','kon_kon', 'kul_rel', 'mil_kli', 'other']
df2y_gr = df_twoYears.groupby('yr')
tab2y = df2y_gr['gender'].value_counts()
n2000, n2015 = tab2y[0] + tab2y[1], tab2y[2] + tab2y[3]
freq = []
freq.append(tab2y[0]/n2000) 
freq.append(tab2y[1]/n2000)
freq.append(tab2y[2]/n2015)
freq.append(tab2y[3]/n2015)


"{:.0%}".format(tab_gen[0])

ubfont = {'fontname':'Calibri'}
fig, ax = plt.subplots()
ax.grid(b=None)
x = tuple(freq)
x1 = "{:.0%}".format(x[1])
x0 = "{:.0%}".format(x[0])
x2 = "{:.0%}".format(x[2])
x3 = "{:.0%}".format(x[3])
width = 0.4
N = 4
ind = np.array([0,0.8])
bar1 = ax.bar(ind[0], x[0], width, color='#772953', edgecolor = "none")
bar2 = ax.bar(ind[0], x[1], width, color='#E95420', edgecolor = "none", bottom = x[0])
bar2 = ax.bar(ind[1], x[2], width, color='#772953', edgecolor = "none")
bar2 = ax.bar(ind[1], x[3], width, color='#E95420', edgecolor = "none", bottom = x[2])
ax.set_xticks(ind + width/2)
ax.set_frame_on(b=False)
ax.set_xticklabels(('2000', '2015'),**ubfont, fontweight='bold')
ax.set_yticklabels([])
ax.set_xlim([-0.2, 1.4])
ax.set_ylim([0, 1])
ax.annotate(x0, (ind[0]+width/2,x[0]-0.05),fontweight='bold',color='white',horizontalalignment='center')
ax.annotate(x1, (ind[0]+width/2,1-x[1]+0.03),fontweight='bold',color='white',horizontalalignment='center')
ax.annotate(x2, (ind[1]+width/2,x[2]-0.05),fontweight='bold',color='white',horizontalalignment='center')
ax.annotate(x3, (ind[1]+width/2,1-x[3]+0.03),fontweight='bold',color='white',horizontalalignment='center')
plt.show()


#%% Plot fra igår mht. emner og køn

#Data til figuren
df_gen = df_mf.groupby(by='gender')
emner = ['int_pol', 'sta_tek', 'fam_id','kon_kon', 'kul_rel', 'mil_kli', 'other']
tab = df_gen.mean().loc[:,emner].T

sum_ = []
for emne in emner:
    ag = df_mf[emne].value_counts()[1]
    sum_.append(ag)

tab['size'] = sum_
x, y = tab['all female'], tab['all male'] 
n = pd.Series(tab.index)
s = 100

##Figuren starter her
fig, ax = plt.subplots()
ax.grid(b=None)
ax.set_frame_on(b=False)
ax.scatter(x, y, s=s, c = '#E95420')
line = mlines.Line2D([0, 1], [0, 1], color='#772953', alpha=0.7)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('Kvinder')
ax.set_ylabel('Mænd')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(0, .5)
ax.set_ylim(0, .5)
for i, txt in enumerate(xlabels):
    ax.annotate(txt, (x[i],y[i]))
#ax.scatter(x, y)
plt.show()

#%% Mænd og kvinder point plot

####################################################
### Figuren er klar til dokumentet #################
####################################################
#Data til figuren
#Data til figuren
df_gen = df_mf.groupby(by='gender')
emner = ['int_pol', 'sta_tek', 'fam_id','kon_kon', 'kul_rel', 'mil_kli']
tab = df_gen.mean().loc[:,emner].T


y_fem = tab['all female']
y_men = tab['all male']
N = len(tab) # giver 7
#y1 = "{:.0%}".format(tab_gen[1])
#y0 = "{:.0%}".format(tab_gen[0])
x = np.array([0,1])
x_men = [0.2]*N
x_fem = [1]*N
ubfont = {'fontname':'Calibri'}

#Husk
## Fonts i dine labels...

labels = ['International','Statsforvaltning og teknologi', 'Familie og identitet','Konflikter og konsekvenser', 'Kultur og religion', 'Miljø og klima', 'Andet']
fig, ax = plt.subplots()
ax.grid(b=None)

#Loop for streger
for i in range(N):
    if y_men[i] > y_fem[i]:
        ax.plot([x_men[i], x_fem[i]], [y_men[i], y_fem[i]], color='#772953', linewidth=3, alpha=1)
        ax.scatter(x_men[i], y_men[i],color='#772953', s=35)
        ax.scatter(x_fem[i], y_fem[i], color='#772953', s = 35)
    else:
        ax.plot([x_men[i], x_fem[i]], [y_men[i], y_fem[i]], color='#E95420', linewidth=3, alpha=1)
        ax.scatter(x_men[i], y_men[i],color='#E95420', s=35)
        ax.scatter(x_fem[i], y_fem[i], color='#E95420', s = 35)
    
ax.set_frame_on(b=False)
ax.set_yticklabels([])
ax.set_xlim([-0.2, 1.2])
ax.set_ylim([0, 0.35])
ax.xaxis.set_ticks(np.array([0.2,1]))
ax.set_xticklabels(('Mænd', 'Kvinder'),**ubfont,fontweight='bold')
for i, txt in enumerate(labels):
    y_men1 = "{:.0%}".format(y_men[i])
    y_fem1 = "{:.0%}".format(y_fem[i])
    if y_men[i] > y_fem[i]:
        ax.annotate(txt + ", " + y_men1, (x_men[i]-0.05,y_men[i]-0.002),color='#772953', horizontalalignment='right', fontweight='bold', **ubfont)
        ax.annotate(y_fem1, (x_fem[i]+0.05,y_fem[i]-0.002),color='#772953', horizontalalignment='left', fontweight='bold', **ubfont)
    else:
        ax.annotate(txt + ", " + y_men1, (x_men[i]-0.05,y_men[i]-0.002),color='#E95420', horizontalalignment='right', fontweight='bold', **ubfont)
        ax.annotate(y_fem1, (x_fem[i]+0.05,y_fem[i]-0.002),color='#E95420', horizontalalignment='left', fontweight='bold', **ubfont)        
plt.show()

#%% Stacked barplot overstørrelsen på paragrafantallet
####################################################
### Figuren er klar til dokumentet #################
####################################################

df_gen = df_mf.groupby(by='gender')
emner = ['int_pol', 'sta_tek', 'fam_id','kon_kon', 'kul_rel', 'mil_kli', 'other']
tab = df_gen.mean().loc[:,emner].T
xlabels = ['International','Statsforvaltning \nog teknologi', 'Familie og \nidentitet','Konflikter og \nkonsekvenser', 'Kultur og \nreligion', 'Miljø og \nklima', 'Andet']

sum_men = []
sum_fem = []
sum_emne = []
for emne in emner:
    fem = (df_gen[emne].value_counts()[1]/len(df_mf))*100
    men = (df_gen[emne].value_counts()[3]/len(df_mf))*100
    sum_fem.append(fem)
    sum_men.append(men)
    sum_emne.append(emne)

tab['fem'], tab['men'] = sum_fem, sum_men
y_men, y_fem = tab['men'], tab['fem'] 
ind = np.arange(len(tab))
width = 0.5
fig, ax = plt.subplots()
ax.grid(b=None)
ax.set_frame_on(b=False)
ax.set_xlim([-0.5,6.5])
ax.xaxis.set_ticks(ind+width/2)
ax.set_xticklabels(xlabels,**ubfont, fontweight='bold')
p1 = ax.bar(ind, y_men, width, color='#772953', edgecolor="none")
p2 = ax.bar(ind, y_fem, width, bottom=y_men , color='#E95420', edgecolor="none")
ax.legend([p1, p2], ['', ''])
ax.set_ylabel('pct.',**ubfont, fontweight='bold')
plt.show()



#%% 
####################################################
### Tidsserie over kønsfordeling   #################
####################################################
tabYr = pd.crosstab(df_mf.yr, df_mf.gender).apply(lambda r: r/r.sum(), axis=1) 
tabYr = tabYr.loc[tabYr.index!=11,:]
x = pd.Series(tabYr.index)
y_men = tabYr['all male']
y_fem = tabYr['all female']



fig, ax = plt.subplots()
ax.set_frame_on(b=False)
ax.grid(b=False)
l1, = ax.plot(x, y_men, color='#772953', linewidth=2)
l2, = ax.plot(x, y_fem, color='#E95420', linewidth=2)
#ax.legend([l1, l2], ['Mænd', 'Kvinder'])
ax.set_ylabel('Andel af spalteplads',**ubfont, fontweight='bold')
plt.show()

     
#%% Kommentarer #Foslag nummer 1
tab = {}
tab.update({'familie':df_mf.loc[df_mf['fam_id']==1,:].groupby('gender').comments_no.mean()}) #God 
tab.update({'statsf.':df_mf.loc[df_mf['sta_tek']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'miljø':df_mf.loc[df_mf['mil_kli']==1,:].groupby('gender').comments_no.mean()}) #God

men =  list([tab['familie']['all male'], tab['statsf.']['all male'], tab['miljø']['all male']])
women = list([tab['familie']['all female'],tab['statsf.']['all female'], tab['miljø']['all female']])

N = len(men)
ind = np.arange(N)
ubfont = {'fontname':'Calibri'}
fig, ax = plt.subplots()
ax.grid(b=None)
width = 0.4
bar1 = ax.bar(ind,  men, width, color='#772953', edgecolor = "none")
bar2 = ax.bar(ind + width, women, width, color='#E95420', edgecolor = "none")
ax.set_xticks(ind + width)
ax.set_frame_on(b=False)
ax.set_xticklabels(('Familie og \nidentitet', 'Statsforvaltning og \nteknologi', 'Miljø og \nklima'),**ubfont, fontweight='bold')
ax.set_yticklabels([])
ax.
#ax.set_xlim([-0.2, 1.4])
#ax.set_ylim([0, 1])
for i in range(N):
    ax.annotate("%.1f" % men[i], (ind[i]+width/2,men[i]-1),fontweight='bold',color='white',horizontalalignment='center', **ubfont)
    ax.annotate("%.1f" % women[i], (ind[i]+3*width/2,women[i]-1),fontweight='bold',color='white',horizontalalignment='center', **ubfont)
plt.show()

#%% Kommentarer #Forslag nummer 1
tab = {}
tab.update({'familie':df_mf.loc[df_mf['fam_id']==1,:].groupby('gender').comments_no.mean()}) #God 
tab.update({'statsf.':df_mf.loc[df_mf['sta_tek']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'miljø':df_mf.loc[df_mf['mil_kli']==1,:].groupby('gender').comments_no.mean()}) #God

men =  list([tab['familie']['all male'], tab['statsf.']['all male'], tab['miljø']['all male']])
women = list([tab['familie']['all female'],tab['statsf.']['all female'], tab['miljø']['all female']])
emner = ['Familie og indentitet', 'Statsforvaltning og teknologi','Miljø og klima']
minorden = [0,1,2]

men = [men[i] for i in minorden]
women = [women[i] for i in minorden]
emner = [emner[i] for i in minorden]


N = len(men)
ind = np.array([0,0.5,1])
ubfont = {'fontname':'Calibri'}
fig, ax = plt.subplots()
#ax.grid(b=None)
#width = 0.4
ax.scatter(men, ind, color='#772953', edgecolor = "none", s=1000, zorder=2)
ax.scatter(women, ind, color='#E95420', edgecolor = "none", s=1000, zorder=2)
#ax.set_xticks(ind + width)
ax.set_frame_on(b=False)
#ax.set_xticklabels(,step = 3 ,**ubfont, fontweight='bold')
ax.set_yticklabels([])
ax.set_xlim([4, 16])
#ax.set_ylim([0, 1])
ax.plot()
for i in range(N):
    ax.plot([men[i], women[i]], [ind[i], ind[i]], color='#AEA79F', linewidth=2, alpha=1, zorder=1)
    if men[i] < women[i]:
        ax.annotate(emner[i], (4, ind[i]),fontweight='bold',color='#333333',horizontalalignment='left', verticalalignment='center',**ubfont)
    else:
        ax.annotate(emner[i], (4, ind[i]),fontweight='bold',color='#333333',horizontalalignment='left', verticalalignment='center', **ubfont)
plt.show()


#%% Plot fra igår mht. emner og køn

#Data til figuren
df_gen = df_mf.groupby(by='gender')
tab = {}
tab.update({'fam_id':df_mf.loc[df_mf['fam_id']==1,:].groupby('gender').comments_no.mean()}) #God 
tab.update({'sta_tek':df_mf.loc[df_mf['sta_tek']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'mil_kli':df_mf.loc[df_mf['mil_kli']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'kul_rel':df_mf.loc[df_mf['kul_rel']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'other':df_mf.loc[df_mf['other']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'kon_kon':df_mf.loc[df_mf['kon_kon']==1,:].groupby('gender').comments_no.mean()}) #God
tab.update({'int_pol':df_mf.loc[df_mf['int_pol']==1,:].groupby('gender').comments_no.mean()}) #God

men = []
women = []
emner = []
wShare = []
for i in list(tab.keys()):
    men.append(tab[i]['all male'])
    women.append(tab[i]['all female'])
    emner.append(i)
    wShare.append((df_gen[i].value_counts()[1]/sum(df_gen[i].value_counts()[[1,3]]))*5005)

N = len(emner)
##Figuren starter her
fig, ax = plt.subplots()
ax.grid(b=None)
ax.set_frame_on(b=False)
for i in range(N):
    if women[i] > men[i]:
        ax.scatter(women[i], men[i], c = '#E95420', s = wShare, edgecolor='none', zorder=2, alpha=0.8)
    else:
        ax.scatter(women[i], men[i], c = '#772953', s = wShare, edgecolor='none', zorder=2, alpha=0.8)        
line = mlines.Line2D([0, 1], [0, 1], color='#333333', alpha=0.7, zorder=1)
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
ax.set_xlabel('Kvinder')
ax.set_ylabel('Mænd')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlim(6, 16)
ax.set_ylim(6, 16)
for i, txt in enumerate(emner):
    ax.annotate(txt, (women[i],men[i]))
#ax.scatter(x, y)
plt.show()

