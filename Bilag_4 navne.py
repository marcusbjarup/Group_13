import pandas as pd 


navne_p = list(pd.read_csv('pige.csv',encoding='utf-8',  header=None)[0])
navne_d = list(pd.read_csv('drenge.csv',encoding='utf-8',  header=None)[0])
#navne_b = list(pd.read_csv('begge.csv',encoding='utf-8',  header=None)[0])
df = pd.read_csv('observationer.csv',encoding='utf-8')
Piger = ['Dominique','Nana','Lykke','Deniz','Kit','Sacha','Pil','Elisa','Maxime','Linn','Mai','Justice','Maria','Nikola','Nour','Nur','Jannie','Robin','Maj','Andrea','Gunde','Gry','Michel','Anda','Misha','Jo', 'Sandy','Rana', 'Anne','Gabi','Bjørk']
Drenge = ['Sam', 'Lave', 'Ray', 'Elias', 'Bo', 'Manu', 'Dan', 'Joan','Tonny','Kim','Tonni','Nadeem','Alex','Ronnie','Addis','Kai','Glenn','Joe','Hamdi','Chris', 'Saman',  'Alaa',   'Roman', 'Sami', 'Dani',    'Benny', 'Iman',  'Ryan',  'Mikka', 'Jean', 'Johnny','Evin']
from itertools import compress
dr1 = [(txt not in Piger) for txt in navne_d]
pi1 = [(txt not in Drenge) for txt in navne_p]
drengenavne = list(compress(navne_d, dr1))
pigenavne = list(compress(navne_p, pi1))
del navne_d, navne_p, dr1, pi1

#%%
def fornavne(names):
    try:
        navne = names.replace(' og ',',')
        navne_mult = navne.split(',')
        navne_mult = list(map(str.strip, navne_mult))
        fornavne = []
        for name in navne_mult:
            fornavne.append(name.split(' ')[0])
    except:
        fornavne = ""
    return fornavne

#%%
def navne_gender(navne):
    first_name = fornavne(navne)
    #print(type(first_name))
    if len(first_name) == 1:
        first_name = first_name[0]
        if first_name in drengenavne:
            gender = 'male'
        elif first_name in pigenavne:
            gender = 'female'
        else:
            gender = 'fejl'
    else:
        n_container = []
        for n in first_name:
            if n in drengenavne:
                n_container.append(0)
            elif n in pigenavne:
                n_container.append(1)
            else:
                pass
        if sum(n_container) == 0:
            gender = 'all male'
        elif sum(n_container) == len(n_container):
            gender = 'all female'
        else:
            gender = 'mixed'
    return gender

#%%
navne = df.loc[:,'names']
df['gender'] = navne.apply(navne_gender)
#%%
df[df['gender']=='mixed']
#%%
n2 = [len(n1[i])> 0 for i in range(len(n1))]
dubletter = n1[n2]
dubs = []
for t in dubletter:
    dubs.append(t[0])

del n1, n2, dubletter

#%%
	for i in range(len(first_name)):
        navn = first_name[i]
        if navn in navne_d:
            if navn in navne_p:
                dublet_navn.append(navn)
            else:
                drenge.append(navn)
        else:
            piger.append(navn)