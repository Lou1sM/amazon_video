import json
import os


def fix(json_dict):
    return {'raw_cap' if k=='raw' else k:v for k,v in json_dict.items()}

for scene_dir in os.listdir('SummScreen/video_scenes'):
    try:
        with open(f'SummScreen/video_scenes/{scene_dir}/swinbert_raw_scene_caps.json') as f:
            d = json.load(f)
    #except IsADirectoryError
    except Exception as e:
        print(scene_dir,e)
        continue
    if len(d) == 0:
        print('d is 0 for',scene_dir)
        continue
    if 'raw' in d[0].keys():
        assert all(['raw' in x.keys() for x in d])
        print(f'fixing {scene_dir}')
        new_d = [fix(x) for x in d]
        with open(f'SummScreen/video_scenes/{scene_dir}/swinbert_raw_scene_caps.json', 'w') as f:
            json.dump(new_d,f)
