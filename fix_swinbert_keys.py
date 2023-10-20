import json
import os


def fix(json_dict):
    return {'raw_cap' if k=='raw_caps' else k:v for k,v in json_dict.items()}

for scene_dir in os.listdir('SummScreen/video_scenes'):
    with open(f'SummScreen/video_scenes/{scene_dir}/swinbert_raw_scene_caps.json') as f:
        d = json.load(f)
    if len(d) == 0:
        breakpoint()
        continue
    if 'raw_caps' in d[0].keys():
        assert all(['raw_caps' in x.keys() for x in d])
        print(f'fixing {scene_dir}')
        new_d = [fix(x) for x in d]
        with open(f'SummScreen/video_scenes/{scene_dir}/swinbert_raw_scene_caps.json', 'w') as f:
            json.dump(new_d,f)
