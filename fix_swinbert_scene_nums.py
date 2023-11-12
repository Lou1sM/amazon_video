import os
import json
from dl_utils.misc import check_dir
from tqdm import tqdm
from episode import episode_from_epname


def num_from_name(en):
    return int(en.split('.')[0].split('_')[1][5:])

no_scenes = []
epnames_with_caps = os.listdir('SummScreen/video_scenes')
for en in tqdm(epnames_with_caps):
    available_scenes=[x for x in os.listdir(f'SummScreen/video_scenes/{en}') if x.endswith('mp4')]
    #available_scenes.sort(key=num_from_name)
    if len(available_scenes)==0:
        no_scenes.append(en)
        continue
    ep = episode_from_epname(en)
    true_n_scenes = len(ep.scenes)
    if len(available_scenes) == true_n_scenes:
        continue # no missing scenes
    else:
        missing_scenes = [sn for sn in range(true_n_scenes) if f'{en}_scene{sn}.mp4' not in available_scenes]
        assert len(missing_scenes) != 0
        for fn in ['swinbert','kosmos']:
            with open(f'SummScreen/video_scenes/{en}/{fn}_raw_scene_caps.json') as f:
                caps = json.load(f)
            assert len(caps)+len(missing_scenes)==true_n_scenes+1
            fixed_caps = []
            j = 0
            for sn in range(true_n_scenes):
                if sn in missing_scenes:
                    cap = to_append = {'raw_cap':''}
                else:
                    to_append = caps[j]
                    j+=1
                to_append['scene_id'] = f'{en}s{sn}'
                fixed_caps.append(to_append)
            assert j==len(caps)
            assert fixed_caps[:missing_scenes[0]]==caps[:missing_scenes[0]]
            check_dir(f'SummScreen/fixed_scene_num_caps/{en}')
            with open(f'SummScreen/fixed_scene_num_caps/{en}/{fn}_raw_scene_caps.json','w') as f:
                json.dump(fixed_caps,f)
