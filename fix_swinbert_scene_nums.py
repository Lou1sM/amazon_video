import os
import json
from dl_utils.misc import check_dir


check_dir('SummScreen/fixed_scene_num_caps/')

def num_from_name(en):
    return int(en.split('.')[0].split('_')[1][5:])

ep_names_with_caps = os.listdir('SummScreen/video_scenes')
for en in ep_names_with_caps:
    print(en)
    available_scenes=[x for x in os.listdir(f'SummScreen/video_scenes/{en}') if x.endswith('mp4')]
    available_scenes.sort(key=num_from_name)
    highest_scene_num = num_from_name(available_scenes[-1])
    if len(available_scenes) == highest_scene_num:
        continue # no missing scenes
    else:
        missing_scenes = [sn for sn in range(highest_scene_num) if f'{en}_scene{sn}.mp4' not in available_scenes]
        for fn in ['swinbert','kosmos']:
            with open(f'SummScreen/video_scenes/{en}/{fn}_raw_scene_caps.json') as f:
                caps = json.load(f)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT

