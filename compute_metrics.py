from datasets import load_dataset
import numpy as np
from nltk.tokenize import sent_tokenize
import torch
from UniEval.utils import convert_to_json
from UniEval.metric.evaluator import get_evaluator
import pandas as pd
import json
from PREFS.factscorer import FactScorer
from dl_utils.misc import check_dir
from utils import rouge_from_multiple_refs, get_all_testnames, is_prisma_wellformed, postfilter
from os.path import join
import argparse
from episode import episode_from_name
import os
from PREFS.atomic_facts import AtomicFactGenerator
from tqdm import tqdm
from summac.model_summac import SummaCConv


def filter_facts(facts):
    """Some facts are vacuous of any real information, exclude these before scoring."""
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or (not any(x in f.lower() for x in ['someone','something','somebody','is a person','is a character', 'are people', 'are characters']) and len(f.split())>2)]
    facts = [f for f in facts if not f.lower().startswith('there is a') and not 'is in a room' in f and not 'is talking' in f and not 'are talking' in f and not 'made a statement' in f]
    facts = [f for f in facts if not 'is mentioned' in f.lower() and not 'are mentioned' in f.lower() and not 'is there' in f.lower() and not 'are there' in f.lower()]
    facts = [f for f in facts if f=='<MALFORMED SENTENCE>' or not f.endswith(' to')]
    facts = [f for f in facts if not 'is there' in f and not 'are there' in f]
    return facts

def get_maybe_cached_atomic_facts(maybe_cached_path, generator, nl_text=None, path=None, no_cache=False):
    assert (nl_text is None) != (path is None)
    if nl_text is None:
        with open(path) as f:
            nl_text = f.read()
    if os.path.exists(maybe_cached_path) and not no_cache:
        print('using extraction cache at', maybe_cached_path)
        with open(maybe_cached_path) as f:
            facts = f.read().split('\n')
    else:
        print('recomputing facts to store in', maybe_cached_path)
        facts_and_sources = generator.convert_to_facts(nl_text)
        facts = [x for line in facts_and_sources for x in line[1]]
        with open(maybe_cached_path,'w') as f:
            f.write('\n'.join([pf for pf in facts]))

    filtered_facts = filter_facts(facts)
    print('filtering:', [x for x in facts if x not in filtered_facts])
    filtered_facts = [' '.join('is' if w=='was' else w for w in f.split()) for f in filtered_facts]
    if ARGS.is_test:
        filtered_facts = filtered_facts[:3]
    return filtered_facts


if __name__ == '__main__':
    generator = AtomicFactGenerator()

    all_metrics = ['prisma', 'rouge', 'summac', 'unieval']
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname',type=str,default='tmp')
    parser.add_argument('--ep',type=str,default='none')
    parser.add_argument('--dset', type=str, default='moviesumm')
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--print-results',action='store_true')
    parser.add_argument('--save-anyway',action='store_true')
    parser.add_argument('--no-prisma-rec-cache',action='store_true')
    parser.add_argument('--no-prisma-prec-cache',action='store_true')
    parser.add_argument('--no-summac-cache',action='store_true')
    parser.add_argument('--no-unieval-cache',action='store_true')
    parser.add_argument('--print-unieval',action='store_true')
    parser.add_argument('--expdir-prefix', type=str, default='experiments')
    parser.add_argument('--metrics', type=str, nargs='+', choices=all_metrics+['all','just-get-facts'], default=['all'])
    parser.add_argument('--n-dpoints', type=int, default=-1)
    parser.add_argument('--start-from', type=int, default=-1)
    ARGS = parser.parse_args()

    if ARGS.metrics == ['all']:
        ARGS.metrics = all_metrics

    fs = FactScorer('gpt-4-turbo-preview', cache_dir_prefix='metrics-caches')


    expdir = join(ARGS.expdir_prefix, ARGS.expname)
    gendir = join(expdir, 'generations_test')

    if ARGS.metrics == ['all']:
        ARGS.metrics = all_metrics
    if 'rouge' in ARGS.metrics:
        ARGS.metrics = [m for m in ARGS.metrics if m!='rouge']+['r1','r2','rl','rlsum']
    if 'prisma' in ARGS.metrics:
        ARGS.metrics.append('n_facts')
        check_dir(prisma_prec_cache_dir :=join('metrics-caches', 'gpt-extracted-facts', ARGS.expname))
        check_dir(prisma_rec_cache_dir:='metrics-caches/gpt-extracted-facts/gt_summ_facts')
        ARGS.metrics += ['prisma-prec', 'prisma-rec']
    if 'summac' in ARGS.metrics:
        summac_model = SummaCConv(models=['vitc'], bins='percentile', granularity='sentence', nli_labels='e', device='cuda', start_file='default', agg='mean')
        check_dir(summac_cache_dir:=join('metrics-caches', 'summac-cache', ARGS.expname))

    usable_vidnames, clean2cl = get_all_testnames()
    if 'unieval' in ARGS.metrics:
        cl2clean = {v:k for k,v in clean2cl.items()}
        ds = load_dataset("rohitsaxena/MovieSum")
        unievaluator = get_evaluator('summarization')
        unieval_submetrics = ['coherence', 'consistency', 'fluency', 'relevance']
        ARGS.metrics += ['unieval-' + m for m in unieval_submetrics]
        check_dir(unieval_cache_dir:=join('metrics-caches', 'unieval-cache', ARGS.expname))

    all_vidnames = [x.removesuffix('-summary.txt') for x in os.listdir(gendir) if x.endswith('-summary.txt')] if ARGS.ep == 'none' else [f'{ARGS.ep}.txt']
    all_vidnames = [x for x in all_vidnames if x in usable_vidnames]
    if ARGS.n_dpoints != -1:
        all_vidnames = all_vidnames[:ARGS.n_dpoints]
    full_results = {m:{} for m in ARGS.metrics}
    all_factscores = []
    all_rouges = []
    #all_vidnames = ['v-for-vendetta_2005'] + all_vidnames
    running_summac = 0
    for i,vn in enumerate(pbar := tqdm(all_vidnames)):
        if i < ARGS.start_from:
            continue
        ep = episode_from_name(vn, False)
        gt_summs = ep.summaries # this is a dict
        good_summ_sources = ('soapcentral_condensed','tvdb','tvmega_recap') if ARGS.dset=='ss3d' else 'moviesumm'
        with open(f'{gendir}/{vn}-summary.txt') as f:
            pred_summ = f.read()

        pred_sents = sent_tokenize(pred_summ)
        if 'ours' in ARGS.expname:
            pred_sents = [s for s in pred_sents if postfilter(s)]
            pred_sents = [s.replace('Lt.', 'Lieutenant').replace('Dr.', 'Doctor') for s in pred_sents if postfilter(s)]
        rw_bridge = [s for s in pred_sents if 'rewritten' in s]
        frac_rewritten = 0
        if len(rw_bridge) == 1:
            rwidx = pred_sents.index(rw_bridge[0])
            frac_rewritten = rwidx/len(pred_sents)
            pred_sents = pred_sents[:rwidx]
        elif len(rw_bridge)>1:
            print('multiple rw-bridges:', rw_bridge)
        else:
            frac_rewritten = 0
        pred_sents = ['<MALFORMED SENTENCE>' if not is_prisma_wellformed(ps) else ps for ps in pred_sents]
        #pred_sents = [ps for ps in pred_sents if is_prisma_wellformed(ps) ]

        # Now compute specified metrics
        if 'prisma-prec' in ARGS.metrics or ARGS.metrics==['just-get-facts']:
            pbar.set_description(f'Computing prisma-prec for {vn}')
            cache_fpath = join(prisma_prec_cache_dir, vn)

            pred_summ_prec = ' '.join(pred_sents)
            pred_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=pred_summ_prec, no_cache=ARGS.no_prisma_prec_cache)
            #pred_facts = [pf for pf in pred_facts if 'MALF' not in pf]
            if 'prisma-prec' in ARGS.metrics:
                vidname_to_use = f'{vn}-gtup' if ARGS.expname == 'gt-upperbound' else vn
                score = fs.get_score(pred_facts,
                                                ref_summaries_dict=gt_summs,
                                                summname=vidname_to_use,
                                                topic='A summary of a TV show',
                                                overwrite_cache=ARGS.no_prisma_prec_cache)
                if not ARGS.expname.startswith('ours'):
                    score = score*(1-frac_rewritten)
                print('frac_rewritten', frac_rewritten)
                full_results['prisma-prec'][vn] = score
                full_results['n_facts'][vn] = len(pred_facts)

        if 'prisma-rec' in ARGS.metrics:
            pbar.set_description(f'Computing prisma-rec for {vn}')
            facts_by_summ_name = {}
            gt_facts = []
            cache_fpath = join(prisma_rec_cache_dir, vn)
            gt_summ = ''
            for k in ('tvmega_recap','moviesumm'):
                if k in ep.summaries:
                    gt_summ = ep.summaries[k]
                    break
            gt_facts = get_maybe_cached_atomic_facts(cache_fpath, generator, nl_text=gt_summ, no_cache=ARGS.no_prisma_rec_cache)

            pred_summ_dict = {'pred':pred_summ}
            if all([ps=='<MALFORMED SENTENCE>' for ps in pred_sents]):
                full_results['prisma-rec'][vn] = 0
            else:
                score = fs.get_score(gt_facts,
                                    ref_summaries_dict=pred_summ_dict,
                                    summname=f'{vn}_{ARGS.expname}',
                                    topic='A summary of a movie',
                                    overwrite_cache=ARGS.no_prisma_rec_cache)
                full_results['prisma-rec'][vn] = score

        if 'r1' in ARGS.metrics:
            pbar.set_description(f'Computing rouge for {vn}')

            rouge_results = rouge_from_multiple_refs(pred_summ,
                                                     gt_summs.values(),
                                                     benchmark_rl=True,
                                                     return_full=False)

            for r,s in zip(['r1','r2','rl','rlsum'], rouge_results):
                full_results[r][vn] = s

        if 'summac' in ARGS.metrics:
            #pbar.set_description(f'Computing summac for {vn}')
            pred_sents_summac = [ps for ps in pred_sents if ps!='<MALFORMED SENTENCE>']
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.startswith('and')]
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.startswith('for')]
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.startswith('m., ')]
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.startswith('camera, ')]
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.startswith('of, ')]
            pred_sents_summac = [ps for ps in pred_sents_summac if not ps.endswith('The.')]
            pred_sents_summac = [ps for ps in pred_sents_summac if 'the link' not in ps]
            pred_sents_summac = [ps for ps in pred_sents_summac if 'more info' not in ps]
            pred_sents_summac = [ps for ps in pred_sents_summac if ' on.' not in ps]
            pred_sents_summac = [ps for ps in pred_sents_summac if 'Suicide Prevention' not in ps]
            pred_sents_summac = [ps for ps in pred_sents_summac if not any(x in ps.lower() for x in ['summaries', 'summary', 'plot point', 'is shown', 'displayed', 'scene', 'english', 'irish', 'british','released', 'concluded', '...', 'scottish','ltl', 'ltl', 'lt-l'])]
            #pred_sents_summac = [ps for ps in pred_sents_summac if 'summary' not in ps]
            #pred_sents_summac = [ps for ps in pred_sents_summac if 'plot point' not in ps]
            #pred_sents_summac = [ps for ps in pred_sents_summac if 'is shown' not in ps]
            #pred_sents_summac = [ps for ps in pred_sents_summac if 'scene' not in ps.lower()]
            #pred_sents_summac = [ps for ps in pred_sents_summac if 'english' not in ps.lower()]
            pred_summ_summac = ' '.join(pred_sents_summac)
            maybe_cached_summac_path = join(summac_cache_dir, vn)
            if os.path.exists(maybe_cached_summac_path) and not ARGS.no_summac_cache:
                with open(maybe_cached_summac_path) as f:
                    full_results['summac'][vn] = float(f.read())
            else:
                best_summac = 0
                if len(pred_sents_summac) > 0:
                    for v in gt_summs.values():
                        new_summac = summac_model.score([v]*len(pred_sents_summac), pred_sents_summac)['scores']
                        for ps,sc in zip(pred_sents_summac, new_summac):
                            if sc > 0.28:
                                print(ps, round(sc,4))
                        new_summac = np.array(new_summac).mean()
                        #new_summac = summac_model.score([v], pred_sents)['scores'][0]
                        if new_summac > best_summac:
                            best_summac = new_summac
                running_summac = (best_summac + i*running_summac)/(i+1)
                pbar.set_description(f'summac: {running_summac:.3f}')
                with open(maybe_cached_summac_path,'w') as f:
                    f.write(str(best_summac))
                full_results['summac'][vn] = best_summac

        if 'unieval' in ARGS.metrics:
            unie_pred_summ_sents = sent_tokenize(pred_summ)
            if os.path.exists(maybe_cached_unieval_path:=join(unieval_cache_dir, vn)) and not ARGS.no_unieval_cache:
                with open(maybe_cached_unieval_path) as f:
                    eval_scores = json.load(f)
            else:
                if ARGS.expname in ['ours', 'ours-masked-names']:
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if not any(x in s.split() for x in ['I', "'I'll","I'm"])]
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if 'let me know' not in s.lower().split()]
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if 'happy to help' not in s.lower().split()]
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if 'for example' not in s.lower().split()]
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if 'you' not in s.lower().split() or '"' in s]
                    unie_pred_summ_sents = [s for s in unie_pred_summ_sents if ('the movie' not in s.lower().split() or 'the movie begins' in s or 'the movie ends' in s or 'the movie opens' or 'the movie finishes' in s)]
                    unie_pred_summ = ' '.join(unie_pred_summ_sents)
                else:
                    unie_pred_summ = pred_summ
                gt_match_name = cl2clean[vn]
                gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
                gt_script = gt_match['script']
                if vn=='hellbound-hellraiser-ii_1988':
                    continue
                    breakpoint()
                if len(gt_script) > 180000:
                    gt_script = gt_script[:50000] + gt_script[-50000:]
                print(len(gt_script))
                while True:
                    try:
                        gt_summ = gt_match['summary']
                        data = convert_to_json(output_list=[unie_pred_summ], src_list=[gt_script], ref_list=[gt_summ])
                        eval_scores = unievaluator.evaluate(data, print_result=ARGS.print_unieval)[0]
                        break
                    except torch.OutOfMemoryError:
                        gt_script = gt_script[:int(len(gt_script)*9/20)] + gt_script[-int(len(gt_script)*9/20):]
                        breakpoint()
                        torch.cuda.empty_cache()
                        print(f'OOM for {vn} unieval, trying again with script len {len(gt_script)}')
                with open(maybe_cached_unieval_path, 'w') as f:
                    json.dump(eval_scores, f)
            for k,v in eval_scores.items():
                if k=='overall':
                    full_results[f'unieval'][vn] = v
                else:
                    full_results[f'unieval-{k}'][vn] = v

        if ARGS.is_test and i==1:
            break
        results_df_tmp = pd.DataFrame(full_results)
        if 'prisma-prec' in ARGS.metrics:
            print('running factscore:', results_df_tmp.mean(axis=0)['prisma-prec'])
        if 'prisma-rec' in ARGS.metrics:
            print('running rev-factscore:', results_df_tmp.mean(axis=0)['prisma-rec'])

    results_df = pd.DataFrame(full_results)
    if 'prisma-rec' in ARGS.metrics and 'prisma-prec' in ARGS.metrics:
        results_df['prisma'] = 2/((1/results_df['prisma-prec']) + (1/results_df['prisma-rec']))
    results_df.loc['mean'] = results_df.mean(axis=0)
    results_df.loc['std'] = results_df.std(axis=0)
    assert set(results_df.columns) == set(ARGS.metrics)

    if ARGS.n_dpoints==-1 or ARGS.save_anyway:
        for m in ARGS.metrics:
            m_results = results_df[m].to_dict()
            check_dir(res_dir := join(expdir,'results_by_metric'))
            with open(join(res_dir, f'{m}-full-results.json'), 'w') as f:
                json.dump(m_results,f)

        results_df.to_csv(join(expdir, 'full_results.csv'))
    if ARGS.print_results:
        print(results_df)
