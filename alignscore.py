from alignscore import AlignScore
from datasets import load_dataset
import json
from tqdm import tqdm
from episode import episode_from_name
from utils import get_all_testnames, postfilter
import argparse
import os
from nltk.tokenize import sent_tokenize
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--expname',type=str, required=True)
parser.add_argument('--ep',type=str,default='none')
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--print-results',action='store_true')
parser.add_argument('--no-cache',action='store_true')
parser.add_argument('--postfilter',action='store_true')
parser.add_argument('--expdir-prefix', type=str, default='experiments')
parser.add_argument('--chkpt', type=str, default='base')
parser.add_argument('--n-dpoints', type=int, default=-1)
parser.add_argument('--bs', type=int, default=16)
ARGS = parser.parse_args()

expdir = os.path.join(ARGS.expdir_prefix, ARGS.expname)
gendir = os.path.join(expdir, 'generations_test')

usable_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}
ds = load_dataset("rohitsaxena/MovieSum")
all_vidnames = [x[:-12] for x in os.listdir(gendir) if x.endswith('-summary.txt')] if ARGS.ep == 'none' else [f'{ARGS.ep}.txt']
all_vidnames = [x for x in all_vidnames if x in usable_vidnames]
#versions = ['nli_sp', 'nli', 'bin_sp', 'bin']
versions = ['bin_sp']
scorers = {v:AlignScore(model='roberta-base', batch_size=ARGS.bs, device='cuda:0', ckpt_path=f'AlignScore/AlignScore-{ARGS.chkpt}.ckpt', verbose=False, evaluation_mode=v) for v in versions}
results = []
running_means = {v:0 for v in versions}
if ARGS.n_dpoints != -1:
    all_vidnames = all_vidnames[:ARGS.n_dpoints]
for i,vn in enumerate(pbar := tqdm(all_vidnames)):

    with open(f'{gendir}/{vn}-summary.txt') as f:
        pred_summ = f.read()
    ep = episode_from_name(vn, False)
    gt_summ = ep.summaries['moviesumm']
    gt_match_name = cl2clean[vn]
    gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
    gt_script = gt_match['script']
    pred_sents = [r for s in sent_tokenize(pred_summ) for r in s.split('\n') if r.strip() != '']
    if ARGS.postfilter:
        pred_sents = [s for s in pred_sents if postfilter(s)]
        pred_sents = [s for s in pred_sents if not any(x in s.lower() for x in ['the plot', 'the movie', 'the scene'])]
    rw_bridge = [s for s in pred_sents if 'rewritten' in s]
    frac_rewritten = 0
    if len(rw_bridge) >= 1:
        rwidx = pred_sents.index(rw_bridge[0])
        frac_rewritten = rwidx/len(pred_sents)
        pred_sents = pred_sents[:rwidx] + rw_bridge
        if len(rw_bridge)>1:
            print('multiple rw-bridges:', rw_bridge)
    else:
        frac_rewritten = 0
    #if 'ours' in ARGS.expname:

    new_results = {}
    for m,s in scorers.items():
        new_scores = s.score(contexts=[gt_summ]*len(pred_sents), claims=pred_sents)
        new_scores = [s*(1-frac_rewritten) for s in new_scores]
        new_results[m] = new_scores
        #new_results[m] = s.score(contexts=[gt_script], claims=[' '.join(pred_sents)])
        new_results[m+'-mean'] = 1 if len(new_scores)==0 else sum(new_scores) / len(new_scores)
        running_means[m] = (new_results[m+'-mean'] + i*running_means[m])/(i+1)

    pbar.set_description('  '.join(f'{k}: {v:.3f}' for k,v in running_means.items()))

    #for s, sc in zip(pred_sents, new_results['nli_sp']):
        #print(f'{s}: {sc:.4f}')
    results.append(new_results)

out_json_path = os.path.join(expdir, 'dict-align-iresults.json')
with open(out_json_path, 'w') as f:
    json.dump(results, f)

global_means = {k:sum(sum([z[k] for z in results],[]))/len(results) for k in versions}
results_df = pd.DataFrame(results).drop(versions, axis=1)
results_df.loc['mean'] = results_df.mean(axis=0)
results_df.loc['std'] = results_df.std(axis=0)
results_df.to_csv(os.path.join(expdir, 'full-align-results.csv'))
print(results_df)
breakpoint()

