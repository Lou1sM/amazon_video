from factscore.factscorer import FactScorer
from episode import episode_from_ep_name
import os


fs = FactScorer(model_name='retrieval+ChatGPT',
                data_dir=".cache/factscore/",
                model_dir=".cache/factscore/",
                cache_dir=".cache/factscore/",
                openai_key='factscore/api.key',
                cost_estimate='consider-cache',
                abstain_detection_type='generic')

#all_ep_names = []
#all_generations = []
#all_ref_summaries = []
#all_atomic_facts = []

for fn in os.listdir('SummScreen/cached_chatgpt_facts/full_pred_summ_facts')[:3]:
    assert fn.endswith('.txt')
    ep_name = fn[:-4]
    #all_ep_names.append(ep_name)

    ep = episode_from_ep_name(ep_name)
    #all_ref_summaries.append(ep.summaries)

    with open(f'SummScreen/cached_chatgpt_facts/full_pred_summ_facts/{fn}') as f:
        atomic_facts = f.readlines()

    out = fs.get_score(topics=[ep_name],
                   #generations=all_generations,
                   ref_summaries=[ep.summaries],
                   atomic_facts=[atomic_facts])

    print(out)
import pdb; pdb.set_trace()  # XXX BREAKPOINT
