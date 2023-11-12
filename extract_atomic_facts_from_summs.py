from factscore.atomic_facts import AtomicFactGenerator
import os
from episode import episode_from_epname


generator = AtomicFactGenerator("factscore/api.key", "factscore/demos", gpt3_cache_file=None)

def get_maybe_cached_atomic_facts(maybe_cached_path, pred=None, path=None):
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

def factscore_eval_ep(epname):
    maybe_cache = f'SummScreen/cached_chatgpt_facts/full_pred_summ_facts/{epname}.txt'
    summ_summ_path = f'SummScreen/full_summs/{epname}.txt'
    #pred_facts = get_maybe_cached_atomic_facts(maybe_cache, path=summ_summ_path)
    with open(summ_summ_path) as f:
        pred = f.read()
    pred_facts = generator.run(pred,maybe_cache,cost_estimate=None)

    ep = episode_from_epname(epname)
    best_iou = -1
    best_fs = None
    for summ_name, gt_summ in ep.summaries.items():
        if len(gt_summ)>5000:
            print(f'too long--{len(gt_summ)}')
            continue
        maybe_cache_gt = f'SummScreen/cached_chatgpt_facts/gt_summ_facts/{epname}-{summ_name}.txt'
        gt_facts = get_maybe_cached_atomic_facts(maybe_cache_gt, pred=gt_summ)
        new_fs = fact_fscore(pred_facts, gt_facts)
        if new_fs['iou'] > best_iou:
            best_fs = new_fs
    print(best_fs)

def fact_fscore(pred_facts, gt_facts):
    tp = len([x for x in pred_facts if x in gt_facts])
    fp = len([x for x in pred_facts if x not in gt_facts])
    fn = len([x for x in gt_facts if x not in pred_facts])
    return {'prec':  tp/(tp+fp), 'rec':  tp/(tp+fn), 'iou': tp/(tp+fp+fn)}


if __name__ == '__main__':
    for fname in os.listdir('SummScreen/full_summs'):
        assert fname.endswith('.txt')
        epname = fname[:-4]
        print(epname)
        factscore_eval_ep(epname)

