import json
import pandas as pd
from episode import episode_from_epname


def get_input_and_output(epname):
    with open(f'SummScreen/transcripts/{epname}.json') as f:
        transcript = json.load(f)
    with open(f'SummScreen/summaries/{epname}.json') as f:
        summary = json.load(f)
    return transcript, summary

dset_info = pd.read_csv('dset_info.csv',index_col=0)
names_to_use = dset_info.index[dset_info['has_summ']]

for split in ('train','val','test'):
    mask = dset_info['usable'] & (dset_info['split']==split)
    names_to_use = dset_info.index[mask]

    dpoints = []
    capped_dpoints = []
    for ntu in names_to_use:
        print(ntu)
        #with open(f'SummScreen/transcripts/{ntu}.json') as f:
            #transcript = '\n'.join(json.load(f)['Transcript'])
        with open(f'SummScreen/summaries/{ntu}.json') as f:
            summary_dict = json.load(f)

        ep = episode_from_epname(ntu)
        with open(f'SummScreen/video_scenes/{ntu}/kosmos_procced_scene_caps.json') as f:
            caps_data = json.load(f)
        cdd = {c['scene_id']:c['with_names'] for c in caps_data}
        capped_tlines = []
        tlines = []
        for i,es in enumerate(ep.scenes):
            cap = cdd.get(f'{ntu}s{i}','')
            capped_tlines.append(cap+es+'\n')
            tlines.append(es+'\n')
        transcript = '\n'.join(tlines)
        capped_transcript = '\n'.join(capped_tlines)
        if split == 'train':
            for k,v in summary_dict.items():
                if '[ RECAP AVAILABLE ]' in v or 'Episode summary coming soon.' in v:
                    continue
                assert (k=='tvmega_summary') == (v.startswith('Episode'))
                if len(v) > 0 and k not in ['soap_central','tvmega_summary']:
                    new_dpoint = {'id':ntu, 'pid':f'{ntu}_0', 'input':transcript, 'output':v}
                    new_capped_dpoint = {'id':ntu, 'pid':f'{ntu}_0', 'input':capped_transcript, 'output':v}
                    assert all(isinstance(x,str) for x in new_dpoint.values())
                    dpoints.append(new_dpoint)
                    capped_dpoints.append(new_capped_dpoint)
        else:
            new_dpoint = dict(summary_dict, transcript=transcript, epname=ntu)
            dpoints.append(new_dpoint)
            new_capped_dpoint = dict(summary_dict, transcript=capped_transcript, epname=ntu)
            capped_dpoints.append(new_capped_dpoint)

    with open(f'SummScreen/baseline_{split}set.json','w') as f:
        json.dump(dpoints, f)
    with open(f'SummScreen/baseline_{split}set_small.json','w') as f:
        json.dump(dpoints[:10], f)

    with open(f'SummScreen/baseline_{split}set.json','w') as f:
        json.dump(capped_dpoints, f)
    with open(f'SummScreen/baseline_{split}set_small.json','w') as f:
        json.dump(capped_dpoints[:10], f)
