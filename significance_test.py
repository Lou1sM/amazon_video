import numpy as np
from scipy.stats import t
import pandas as pd


def basic_ttest(mu1,mu2,var1,var2,n1,n2):
    #var1_unbiased = sig1**2*n/(n-1) pandas does this automatically
    #var2_unbiased = sig2**2*n/(n-1)
    #sp = ((var1_unbiased + var2_unbiased)/2)**.5
    pooled_var = (((n1-1)*var1) + ((n2-1)*var2))/(n1+n2-2)
    denom = (pooled_var*(1/n1)*(1/n2))**0.5
    t_val = (mu1-mu2)/denom
    breakpoint()
    print(t_val)
    n = n1+n2-2
    if t_val>t.ppf(.99, df=n-1):
        print(.99)
    elif t_val>t.ppf(.95, df=n-1):
        print(.95)
    elif t_val>t.ppf(.9, df=n-1):
        print(.9)
    elif t_val>t.ppf(.8, df=n-1):
        print(.8)
    return t_val

def stats_from(results_df):
    n = results_df['n_facts'].sum()
    num_correct = (results_df['n_facts']*results_df['factscore']).sum()
    p = num_correct/n
    var = p*(1-p)
    print(p,var,n)
    return p, var, n


def significance(df1, df2):
    p1, var1, n1 = stats_from(df1)
    p2, var2, n2 = stats_from(df2)
    return basic_ttest(p1, p2, var1, var2, n1, n2)
    #pooled_var = (((n1-1)*var1) + ((n2-1)*var2))/(n1+n2-2)
    #denom = (pooled_var*(1/n1)*(1/n2))**0.5
    #t = (p1-p2)/denom
    #return t


ours = pd.read_csv('experiments/kosmos_reordered2/full_results.csv',index_col=0)
unl = pd.read_csv('experiments/unl1-long/full_results.csv',index_col=0)
print(significance(ours, unl))



res = pd.read_csv('rouge_factscore_results.csv', index_col=0)
for metric in res.columns:
    mu1, sig1 = res.loc['kosmos_reordered',metric][:-1].split(' (')
    mu2, sig2 = res.loc['nocaptions',metric][:-1].split(' (')
    mu1,sig1,mu2,sig2 = float(mu1),float(sig1),float(mu2),float(sig2)
    print(f'\n{metric}')
    basic_ttest(mu1,mu2,sig1,sig2,5,5)
