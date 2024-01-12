import numpy as np
from scipy.stats import t
import pandas as pd


def basic_ttest(mu1,sig1,mu2,sig2,n):
    var1_unbiased = sig1**2*n/(n-1)
    var2_unbiased = sig2**2*n/(n-1)
    sp = ((var1_unbiased + var2_unbiased)/2)**.5
    denom = sp*((2/n)**.5)
    t_val = (mu1-mu2)/denom
    print(t_val)
    if t_val>t.ppf(.99, df=n-1):
        print(.99)
    elif t_val>t.ppf(.95, df=n-1):
        print(.95)
    elif t_val>t.ppf(.9, df=n-1):
        print(.9)
    elif t_val>t.ppf(.8, df=n-1):
        print(.8)

res = pd.read_csv('rouge_factscore_results.csv', index_col=0)
for metric in res.columns:
    mu1, sig1 = res.loc['kosmos_reordered',metric][:-1].split(' (')
    mu2, sig2 = res.loc['nocaptions',metric][:-1].split(' (')
    mu1,sig1,mu2,sig2 = float(mu1),float(sig1),float(mu2),float(sig2)
    print(f'\n{metric}')
    basic_ttest(mu1,sig1,mu2,sig2,5)
