import os
import json
from dl_utils.misc import check_dir
from tqdm import tqdm
from episode import episode_from_ep_name


def num_from_name(en):
    return int(en.split('.')[0].split('_')[1][5:])

no_scenes = []
ep_names_with_caps = os.listdir('SummScreen/video_scenes')
for en in tqdm(ep_names_with_caps):
    for fn in ['swinbert','kosmos']:
        with open(f'SummScreen/video_scenes/{en}/{fn}_raw_scene_caps.json') as f:
            caps = json.load(f)
        ep = episode_from_ep_name(en)
        true_n_scenes = len(ep.scenes)
        if len(caps) == true_n_scenes:
            continue # no missing scenes
        else:
            caps_scenes = [x['scene_id'] for x in caps]
            missing_scenes = [sn for sn in range(true_n_scenes) if f'{en}s{sn}' not in caps_scenes]
            assert missing_scenes == list(range(len(caps),true_n_scenes)),'should only be last ones'
            for m in missing_scenes:
                caps.append({'scene_id':f'{en}s{m}', 'raw_cap':''})
            check_dir(f'SummScreen/fixed_scene_num_caps2/{en}')
            with open(f'SummScreen/fixed_scene_num_caps2/{en}/{fn}_raw_scene_caps.json','w') as f:
                json.dump(caps,f)
