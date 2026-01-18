from caption_each_scene import filter_single_caption
from episode import episode_from_name
import os
import json


in_dir = '/mnt/hrs/video-captions'
out_dir = '/mnt/hrs/postprocessed-video-captions'

for movie_name in os.listdir(in_dir):
    print(movie_name)
    out_caps = []
    try:
        ep = episode_from_name(movie_name)
        transcript = ep.transcript
    except FileNotFoundError as e:
        print(e)
        transcript = []
    with open(os.path.join(in_dir, movie_name, 'llava_procced_scene_caps.json')) as f:
        caps = json.load(f)
    for sc in caps:
        if sc['scene_id'].endswith('.npy'):
            continue
        ncaps = filter_single_caption(sc['raw_cap'], transcript)
        out_caps.append(dict(**sc, with_names=ncaps))
    os.makedirs(os.path.join(out_dir, movie_name), exist_ok=True)
    with open(os.path.join(out_dir, movie_name, 'llava_procced_scene_caps.json'), 'w') as f:
        json.dump(caps, f)

