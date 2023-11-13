import numpy as np
import json
from factscore.factscorer import FactScorer
from dl_utils.misc import check_dir
from utils import rouge_from_multiple_refs
from os.path import join
import argparse
from episode import episode_from_epname
import os
from factscore.atomic_facts import AtomicFactGenerator


def get_maybe_cached_atomic_facts(maybe_cached_path, generator, pred=None, path=None):
    assert (pred is None) != (path is None)
    if os.path.exists(maybe_cached_path):
        with open(maybe_cached_path) as f:
            pred_facts = f.readlines()
    else:
        if pred is None:
            with open(path) as f:
                pred = f.read()
        pred_facts_and_sources, para_breaks = generator.run(pred)
        pred_facts = [x for line in pred_facts_and_sources for x in line[1]]
        with open(maybe_cached_path,'w') as f:
            f.write('\n'.join([pf for pf in pred_facts]))

    return pred_facts

if __name__ == '__main__':
    generator = AtomicFactGenerator("factscore/api.key", "factscore/demos", gpt3_cache_file=None)

    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',type=str,default='tmp')
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    ARGS = parser.parse_args()


    fs = FactScorer(model_name='retrieval+llama+npm',
                    data_dir=".cache/factscore/",
                    model_dir="huggyllama",
                    cache_dir=".cache/factscore/",
                    openai_key='factscore/api.key',
                    cost_estimate='consider-cache',
                    abstain_detection_type='generic')


    gendir = join(expdir:=join('experiments',ARGS.expname), 'generations')
    all_epnames = os.listdir(gendir)
    selected_first = 'oltl-10-18-10.txt'
    if selected_first in all_epnames:
        all_epnames.remove(selected_first)
        all_epnames.insert(0,selected_first)
    full_results = {}
    all_factscores = []
    all_rouges = []
    for fn in all_epnames[:1]:
        assert fn.endswith('.txt')
        epname = fn[:-4]

        ep = episode_from_epname(epname)

        check_dir(cache_dir := 'SummScreen/cached_chatgpt_facts/full_pred_summ_facts')
        cache_fpath = f'{cache_dir}/{epname}'
        summ_fpath = f'{gendir}/{epname}.txt'
        with open(summ_fpath) as f:
            pred = f.read()
        atomic_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, pred=pred)

        factscore = fs.get_score(topics=['A summary of a TV show'],
                       ref_summaries=[ep.summaries],
                       atomic_facts=[atomic_facts[:2]])

        rouge_score = rouge_from_multiple_refs(pred, ep.summaries.values(), benchmark_rl=True, return_fll=False)
        full_results[epname] = {'factscore':factscore, 'rouge':rouge_score}
        all_factscores.append(factscore['score'])
        all_rouges.append(rouge_score)
    rouges_arr = np.array(all_rouges).mean(axis=0)
    mean_factscore = np.array(all_factscores).mean()

    with open(join(expdir, 'results.txt'), 'w') as f:
        f.write(f'Expname: {ARGS.expname}\nFActScore: {mean_factscore:.4f}\n')
        for r,s in zip(['1','2','L'], rouges_arr):
            f.write(f'rouge{r}: {s}\n')

    with open(join(expdir,'full_results.json'),'w') as f:
        json.dump(full_results, f)
