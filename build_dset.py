import json
import os
from random import shuffle
from summarize_dialogue import SoapSummer
from episode import episode_from_ep_name
from tqdm import tqdm


def dpoints_from_ep_names(ep_name_list, scene_caps, do_reorder):
    assert not any(['reordered' in x for x in ep_name_list])
    data_list = []
    summer = SoapSummer(None, None, None, None, caps=scene_caps, do_reorder=do_reorder)
    summ_dir = 'SummScreen/summaries'
    for ep_name in tqdm(ep_name_list):
        ep = episode_from_ep_name(ep_name)
        ss = ''.join(summer.get_scene_summs(ep))
        with open(os.path.join(summ_dir, f'{ep_name}.json')) as f:
            d = json.load(f)
        for k,v in d.items():
            if '[ RECAP AVAILABLE ]' in v or 'Episode summary coming soon.' in v:
                continue
            assert (k=='tvmega_summary') == (v.startswith('Episode'))
            if len(v) > 0 and k not in ['soap_central','tvmega_summary']:
                data_list.append({'scene_summs':ss, 'summ':v, 'summ_name':k, 'ep_name':ep_name})
    return data_list


def build_dset(scene_caps, do_reorder):
    fn = f'{scene_caps}_reordered' if do_reorder else scene_caps
    ep_names = [x for x in os.listdir('SummScreen/video_scenes') if x in os.listdir('SummScreen/keyframes')]
    #ep_names = [x.replace('_reordered','') for x in os.listdir(scene_summ_dir)]
    #assert all([x.endswith('.txt') for x in ep_names])
    assert all([os.path.isdir(f'SummScreen/video_scenes/{x}') for x in ep_names])
    #ep_names = [x[:-4] for x in ep_names]
    shuffle(ep_names)
    train_up_to_idx = int(9*len(ep_names)/10)
    train_ep_names = ep_names[:train_up_to_idx]
    test_ep_names = ep_names[train_up_to_idx:]
    print('getting scene summs for train set')
    train_data_list = dpoints_from_ep_names(train_ep_names, scene_caps, do_reorder)
    print(test_ep_names)
    with open(f'SummScreen/json_datasets/train_{fn}_dset.json','w') as f:
        json.dump(train_data_list, f)

    print('getting scene summs for test set')
    test_data_list = dpoints_from_ep_names(test_ep_names, scene_caps, do_reorder)
    with open(f'SummScreen/json_datasets/test_{fn}_dset.json','w') as f:
        json.dump(test_data_list, f)
    #if i==0:
        #del dpoint['tvmega_recap']
