import numpy as np
import pandas as pd
from evaluate import load
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


def get_maybe_cached_atomic_facts(maybe_cached_path, generator, nl_text=None, path=None):
    assert (nl_text is None) != (path is None)
    if os.path.exists(maybe_cached_path):
        with open(maybe_cached_path) as f:
            pred_facts = f.readlines()
    else:
        if nl_text is None:
            with open(path) as f:
                nl_text = f.read()
        pred_facts_and_sources, para_breaks = generator.run(nl_text)
        pred_facts = [x for line in pred_facts_and_sources for x in line[1]]
        with open(maybe_cached_path,'w') as f:
            f.write('\n'.join([pf for pf in pred_facts]))

    return pred_facts


def get_factscore(facts, knowledge_base_text, epname): #epname need for caching
    factscore = fs.get_score(epname, topics=['A summary of a TV show'],
                  ref_summaries=[knowledge_base_text],
                  atomic_facts=[facts])

    factscore['decisions'] = [{k:int(v) if v in [True,False] else v for k,v in dec.items()} for dec in factscore['decisions'][0]]
    return factscore

if __name__ == '__main__':
    generator = AtomicFactGenerator("factscore/api.key", "factscore/demos", gpt3_cache_file=None)

    #all_metrics = ['factscore', 'rev-factscore', 'rouge', 'bertscore']
    all_metrics = ['factscore', 'rouge', 'bertscore']
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',type=str,default='tmp')
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--allow_bs_cache',action='store_true')
    parser.add_argument('--print_results',action='store_true')
    parser.add_argument('--llama_size', type=str)
    parser.add_argument('--expdir_prefix', type=str, default='experiments')
    parser.add_argument('--metrics', type=str, nargs='+', choices=all_metrics+['all','just-get-facts'], default=['all'])
    ARGS = parser.parse_args()
    #assert ARGS.llama_size in ('7B', '13B', '70B')

    llama_size = '7B' if ARGS.is_test else '13B'
    fs = FactScorer(model_name=f'retrieval+llama{llama_size}+npm',
                    data_dir=".cache/factscore/",
                    model_dir="huggyllama",
                    cache_dir=".cache/factscore/",
                    openai_key='factscore/api.key',
                    cost_estimate='consider-cache',
                    abstain_detection_type='generic')


    #expdir = f'/rds/user/co-maho1/hpc-work/experiments/{ARGS.expname}'
    #expdir = f'experiments/{ARGS.expname}'
    expdir = join(ARGS.expdir_prefix, ARGS.expname)
    gendir = join(expdir, 'generations_test')

    if ARGS.metrics == ['all']:
        ARGS.metrics = all_metrics
    if 'rouge' in ARGS.metrics:
        ARGS.metrics = [m for m in ARGS.metrics if m!='rouge']+['r1','r2','rl','rlsum']
    if 'factscore' in ARGS.metrics:
        ARGS.metrics.append('n_facts')
    if 'bertscore' in ARGS.metrics:
        bertscore = load("bertscore")
        ARGS.metrics = [m for m in ARGS.metrics if m!='bertscore']+['bs-precision','bs-recall','bs-f1']
        check_dir(bs_cache_dir := join(expdir,'bertscore_cache'))

    all_epnames = os.listdir(gendir)
    full_results = {m:{} for m in ARGS.metrics}
    all_factscores = []
    all_rouges = []
    for i,fn in enumerate(pbar := tqdm(all_epnames)):
        assert fn.endswith('.txt')
        epname = fn[:-4]

        ep = episode_from_epname(epname, False)
        gt_summs = ep.summaries # this is a dict

        with open(f'{gendir}/{epname}.txt') as f:
            pred_summ = f.read()


        # Now compute specified metrics
        if 'factscore' in ARGS.metrics or ARGS.metrics==['just-get-facts']:
            pbar.set_description(f'Computing factscore for {epname}')
            check_dir(cache_dir := f'../gpt-3.5-turbo-instruct-facts/{ARGS.expname}')
            cache_fpath = f'{cache_dir}/{epname}'
            pred_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=pred_summ)
            breakpoint()
            if 'factscore' in ARGS.metrics:
                fs_results = get_factscore(pred_facts, gt_summs, epname)
                full_results['factscore'][epname] = fs_results['score']
                full_results['n_facts'][epname] = len(pred_facts)

        if 'rev-factscore' in ARGS.metrics:
            pbar.set_description(f'Computing rev-factscore for {epname}')
            check_dir(cache_dir := '../gpt-3.5-turbo-instruct-facts/gt_summ_facts')
            facts_by_summ_name = {}
            gt_facts = []
            for k,v in gt_summs.items():
                cache_fpath = f'{cache_dir}/{epname}-{k}'
                new_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=v)
                gt_facts += new_facts
            gt_facts = list(set(gt_facts))
            rfs_results = get_factscore(gt_facts, {'pred':pred_summ}, f'{epname}_{ARGS.expname}')
            full_results['rev-factscore'][epname] = rfs_results['score']

        if 'r1' in ARGS.metrics:
            pbar.set_description(f'Computing rouge for {epname}')
            rouge_results = rouge_from_multiple_refs(pred_summ, gt_summs.values(), benchmark_rl=True, return_full=False)
            for r,s in zip(['r1','r2','rl','rlsum'], rouge_results):
                full_results[r][epname] = s

        if 'bs-f1' in ARGS.metrics:
            pbar.set_description(f'Computing bertscore for {epname}')
            maybe_bs_cache_path = join(bs_cache_dir,f'{epname}.json')
            if os.path.exists(maybe_bs_cache_path) and ARGS.allow_bs_cache:
                with open(maybe_bs_cache_path) as f:
                    all_bss = json.load(f)
            else:
                all_bss = bertscore.compute(predictions=[pred_summ]*len(gt_summs), references=list(gt_summs.values()), lang="en", idf=True)
                with open(maybe_bs_cache_path,'w') as f:
                    json.dump(all_bss,f)
            summ_idx_to_pick = np.array(all_bss['f1']).argmax()
            for k,v in all_bss.items():
                if k!='hashcode':
                    full_results[f'bs-{k}'][epname] = v[summ_idx_to_pick]

        if ARGS.is_test and i==1:
            break

    results_df = pd.DataFrame(full_results)
    results_df.loc['mean'] = results_df.mean(axis=0)
    results_df.loc['std'] = results_df.std(axis=0)
    assert set(results_df.columns) == set(ARGS.metrics)

    for m in ARGS.metrics:
        m_results = results_df[m].to_dict()
        #v['mean'] = sum(v.values())/len(v)
        #v['std'] = (sum((x-v['mean'])**2 for x in v.values())/len(v))**.5
        check_dir(res_dir := join(expdir,'results_by_metric'))
        with open(join(res_dir, f'{m}-full-results.json'), 'w') as f:
            json.dump(m_results,f)

    results_df.to_csv(join(expdir, 'full_results.csv'))
    if ARGS.print_results:
        print(results_df)
