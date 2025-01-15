import ffmpeg
import os
from natsort import natsorted
import json
import numpy as np


def get_duration(file_path):
    probe = ffmpeg.probe(file_path)
    duration = float(probe['format']['duration'])
    return duration

with open('tvqa-durs.json') as f:
    existing_info = json.load(f)

def cap_and_listify(splits, end):
    new_splits = [0]
    for s in splits:
        if s-new_splits[-1] > 20:
            new_splits.append(s)
    return new_splits + [end]

info = {}
for show_name in os.listdir('data/full-videos/tvqa'):
    show_info = {}
    for season in natsorted(os.listdir(f'data/full-videos/tvqa/{show_name}')):
        season_info = {}
        for fn in natsorted(os.listdir(f'data/full-videos/tvqa/{show_name}/{season}')):
            vid_path = f'data/full-videos/tvqa/{show_name}/{season}/{fn}'
            ep = fn.removesuffix('.mp4')
            #dur = get_duration(vid_path)
            dur = existing_info[show_name][season][ep]
            our_splits = np.load(f'data/ffmpeg-keyframes-by-scene/tvqa/{show_name}/{season}/{ep}/scenesplit_timepoints.npy')
            psd_splits = np.load(f'data/ffmpeg-keyframes-by-scene/tvqa/{show_name}/{season}/{ep}/psd_split.npy')
            unif_splits = np.arange(0, dur, 180)
            our_splits = cap_and_listify(our_splits, dur)
            psd_splits = cap_and_listify(psd_splits, dur)
            unif_splits = cap_and_listify(unif_splits, dur)
            season_info[ep] = {'duration':dur, 'ours':our_splits, 'psd':psd_splits, 'unif': unif_splits}
        show_info[season] = season_info
    info[show_name] = show_info

with open('tvqa-splits.json', 'w') as f:
    json.dump(info, f)
