import pandas as pd
import numpy as np
import os
import json


full_results_dict = {}
ct_results_dict = {}
def add_results_from(base_expname):
    setting_results_dict = {}
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
        new_results_dir = f'/rds/user/co-maho1/hpc-work/experiments/{expname}/results_by_metric'
        #for mname in ['r1','r2','rlsum','bs-precision','bs-recall','bs-f1']:
        for mname in ['bs-precision','bs-recall','bs-f1']:
            new_results_path = os.path.join(new_results_dir,f'{mname}-full-results.json')
            if os.path.exists(new_results_path):
                #print('getting results from', new_results_path)
                with open(new_results_path) as f:
                    en_results[mname] = json.load(f)['mean']*100
                    #en_results[mname] = json.load(f)['std']
            else:
                en_results[mname] = -1


        full_results_dict[expname] = en_results
        setting_results_dict[expname] = en_results

    setting_results = pd.DataFrame(setting_results_dict).T
    setting_means = setting_results.mask(setting_results.eq(-1)).mean()
    setting_stds = setting_results.mask(setting_results.eq(-1)).std()
    print(setting_results)
    #if (setting_results.drop('factscore',axis=1)==-1).any().any():
        #breakpoint()
    ct_results_dict[base_expname] = {k:f'{v:.2f} ({setting_stds[k]:.2f})' for k,v in setting_means.items()}
    #ct_results_dict[base_expname] = setting_means

add_results_from('unifbreaks')
add_results_from('kosmosonly')
add_results_from('unl')
add_results_from('unllong')
add_results_from('llama')
add_results_from('mistral')
add_results_from('central')
add_results_from('startend')
for caps in ['nocaptions', 'swinbert', 'kosmos']:
    for order in ['', '_reordered']:
        setting_results_list = []
        add_results_from(f'{caps}{order}')
full_df = pd.DataFrame(full_results_dict).T
ct_df = pd.DataFrame(ct_results_dict).T
ct_df = ct_df.rename({'kosmos_reordered':'modular-kosmos (ours)', 'swinbert_reordered':'modular-swinbert (ours)','nocaptions_reordered':'w/o video'})
print(ct_df)
breakpoint()
print(ct_df.drop(['swinbert','nocaptions']).to_latex(float_format="{{:0.2f}}".format))
