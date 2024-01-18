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
from tqdm import tqdm


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
    parser.add_argument('--run_factscore',action='store_true')
    parser.add_argument('--only_get_facts',action='store_true')
    parser.add_argument('--llama_size', type=str)
    ARGS = parser.parse_args()
    assert ARGS.llama_size in ('7B', '13B', '70B')


    fs = FactScorer(model_name=f'retrieval+llama{ARGS.llama_size}+npm',
                    data_dir=".cache/factscore/",
                    model_dir="huggyllama",
                    cache_dir=".cache/factscore/",
                    openai_key='factscore/api.key',
                    cost_estimate='consider-cache',
                    abstain_detection_type='generic')


    #gendir = join(expdir:=join('experiments',ARGS.expname), 'generations_test')
    #expdir = f'/rds/user/co-maho1/hpc-work/experiments/{ARGS.expname}'
    expdir = f'experiments/{ARGS.expname}'
    gendir = join(expdir, 'generations_test')
    all_epnames = os.listdir(gendir)
    full_results = {}
    all_factscores = []
    all_rouges = []
    for fn in tqdm(all_epnames):
        assert fn.endswith('.txt')
        epname = fn[:-4]

        ep = episode_from_epname(epname)

        #check_dir(cache_dir := 'SummScreen/cached_llama_facts/full_pred_summ_facts')
        check_dir(cache_dir := f'../gpt-3.5-turbo-instruct-facts/{ARGS.expname}')
        cache_fpath = f'{cache_dir}/{epname}'
        summ_fpath = f'{gendir}/{epname}.txt'
        with open(summ_fpath) as f:
            pred = f.read()
        atomic_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, pred=pred)

        if not ARGS.only_get_facts:
            factscore = fs.get_score(epname, topics=['A summary of a TV show'],
                          ref_summaries=[ep.summaries],
                          atomic_facts=[atomic_facts])

            factscore['decisions'] = [{k:int(v) if v in [True,False] else v for k,v in dec.items()} for dec in factscore['decisions'][0]]
            rouge_score = rouge_from_multiple_refs(pred, ep.summaries.values(), benchmark_rl=True, return_full=False)
            full_results[epname] = {'factscore':factscore, 'rouge':rouge_score}
            all_factscores.append(factscore['score'])
            all_rouges.append(rouge_score)
        if ARGS.is_test:
            break
    if not ARGS.only_get_facts:
        rouges_arr = np.array(all_rouges).mean(axis=0)
        mean_factscore = np.array(all_factscores).mean()

        with open(join(expdir, 'rouge_and_factscore_results.txt'), 'w') as f:
            f.write(f'Expname: {ARGS.expname}\nFActScore: {mean_factscore:.4f}\n')
            f.write(f'Expname: {ARGS.expname}\n')
            for r,s in zip(['1','2','L','Lsum'], rouges_arr):
                f.write(f'rouge{r}: {s}\n')

        with open(join(expdir,'full_rouge_and_factscore_results.json'),'w') as f:
            json.dump(full_results, f)
