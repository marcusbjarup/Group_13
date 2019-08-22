import pandas as pd
import numpy as np
#%%
def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
    
    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.


if __name__ == '__main__': 
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data = (
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3", # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *", # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4", # coder C
    )

    missing = '*' # indicator for missing values
    array = [d.split() for d in data]  # convert to 2D list of string items
    
    print("nominal metric: %.3f" % krippendorff_alpha(array, nominal_metric, missing_items=missing))
print("interval metric: %.3f" % krippendorff_alpha(array, interval_metric, missing_items=missing))


#%%
df = pd.read_csv('/home/polichinel/Dropbox/KU/7.semester/SDS/DF_overlap.csv')
#%%
int_p = df[['s_international_politik','j_international_politik','international_pol','n_international_politik']].fillna('*')
int_pa = np.array(int_p)
int_pat = np.transpose(int_pa, axes=None)
print(krippendorff_alpha(int_pat, nominal_metric, missing_items=missing))
# Niels er dårlig....
print(int_p.corr())
#%%
# TEST!
print(df[['s_international_politik','j_international_politik','international_pol','n_international_politik']].corr())
# Snedigt!
#%%
km = df[['n_miljø_klima','s_miljø_klima','j_miljø_klima','miljø_klima']].fillna('*')
km_a = np.array(km)
km_at = np.transpose(km_a, axes=None)
print(krippendorff_alpha(km_at, nominal_metric, missing_items=missing))
#%%
stat = df[['n_statsforv_tek','s_statsforv_tek','j_statsforv_tek','statsforv_tek']].fillna('*')
stat_a = np.array(stat)
stat_at = np.transpose(stat_a, axes=None)
print(krippendorff_alpha(stat_at, nominal_metric, missing_items=missing))
#%%
# TEST!
print(df[['n_statsforv_tek','s_statsforv_tek','j_statsforv_tek','statsforv_tek']].corr())
# Snedigt!
#%%
kk = df[['n_Konflikter_konsekvenser','s_Konflikter_konsekvenser','j_Konflikter_konsekvenser','Konflikter_konse']].fillna('*')
kk_a = np.array(kk)
kk_at = np.transpose(kk_a, axes=None)
print(krippendorff_alpha(kk_at, nominal_metric, missing_items=missing))
#%%
krm = df[['n_Kultur_rel_medier','s_Kultur_rel_medier','j_Kultur_rel_medier','Kultur_rel_medier']].fillna('*')
krm_a = np.array(krm)
krm_at = np.transpose(krm_a, axes=None)
print(krippendorff_alpha(krm_at, nominal_metric, missing_items=missing))
#%%
fi = df[['n_familie_identitet','s_familie_identitet','j_familie_identitet','familie_identitet']].fillna('*')
fi_a = np.array(fi)
fi_at = np.transpose(fi_a, axes=None)
print(krippendorff_alpha(fi_at, nominal_metric, missing_items=missing))
#%%
andt = df[['n_andet','s_andet','j_andet','andet']].fillna('*')
andt_a = np.array(andt)
andt_at = np.transpose(andt_a, axes=None)
print(krippendorff_alpha(andt_at, nominal_metric, missing_items=missing))
#%%
ig = df[['n_ignore','s_ignore','j_ignore','ignore']].fillna('*')
ig_a = np.array(ig)
ig_at = np.transpose(ig_a, axes=None)
print(krippendorff_alpha(ig_at, nominal_metric, missing_items=missing))
#%%
s = df[['sum_n','sum_s','sum_j','sum_e']].fillna('*')
s_a = np.array(s)
s_at = np.transpose(s_a, axes=None)
print(krippendorff_alpha(s_at, nominal_metric, missing_items=missing))