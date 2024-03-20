import pandas as pd
import numpy as np
import os


full_results_dict = {}
ct_results_dict = {}
def add_results_from(base_expname):
    for run in range(5):
        expname = f'{base_expname}{run}'
        print(expname)
        results_path = f'experiments/{expname}/results.txt'
        with open(results_path) as f:
            rlines = f.readlines()
        assert rlines[0].startswith('r1: ')
        r1val = float(rlines[0][4:])
        assert rlines[1].startswith('r2: ')
        r2val = float(rlines[1][4:])
        assert rlines[2].startswith('rL: ')
        rlval = float(rlines[2][4:])
        assert rlines[3].startswith('rLsum: ')
        rlsumval = float(rlines[3][7:])

        factscore_results_path = f'experiments/{expname}/rouge_and_factscore_results.txt'
        if not os.path.exists(factscore_results_path):
            factscore = -1
        else:
            with open(factscore_results_path) as f:
                flines = f.readlines()
            assert flines[1].startswith('FActScore: ')
            factscore = float(flines[1][11:])
            assert flines[3].startswith('rouge1: ')
            r1 = float(flines[3][8:])
            assert flines[4].startswith('rouge2: ')
            r2 = float(flines[4][8:])
            assert flines[5].startswith('rougeL: ')
            rl = float(flines[5][8:])
            assert flines[6].startswith('rougeLsum: ')
            rlsum = float(flines[6][11:])

        full_results_dict[expname] = {'r1':r1, 'r2':r2, 'rl': rl, 'rlsum': rlsum, 'r1val':r1val, 'r2val':r2val, 'rlval': rlval, 'rlsumval': rlsumval, 'factscore': factscore}
        setting_results_list.append([r1,r2,rl,rlsum,factscore])

    setting_results = np.array(setting_results_list)
    #setting_means = setting_results.mean(axis=0)
    setting_means = np.nanmean(np.where(setting_results!=-1,setting_results,np.nan),0)
    #setting_stds = setting_results.std(axis=0)
    setting_stds = np.nanstd(np.where(setting_results!=-1,setting_results,np.nan),0)
    ct_results_dict[base_expname] = {k:f'{m:.3f} ({s:.3f})' for k,m,s in zip(['r1','r2','rl','rlsum','factscore'],setting_means, setting_stds)}

for caps in ['nocaptions', 'swinbert', 'kosmos']:
    for order in ['', '_reordered', '_randordered']:
        setting_results_list = []
        add_results_from(f'{caps}{order}')
add_results_from('unifbreaks')
full_df = pd.DataFrame(full_results_dict).T
ct_df = pd.DataFrame(ct_results_dict).T
print(ct_df)
breakpoint()
