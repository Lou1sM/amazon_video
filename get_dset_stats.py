import os
import numpy as np
import json
import pandas as pd
from episode import infer_scene_splits, episode_from_epname
import ffmpeg
from dl_utils.misc import asMinutes
from tqdm import tqdm


all_epnames = [x.split('.')[0] for x in os.listdir('SummScreen/summaries')]
all_epnames.remove('atwt-06-29-01')
all_epnames.insert(0, 'atwt-06-29-01')

dset_stats = {}
#cur_df = pd.read_csv('SummScreen/dset_info.csv', index_col=0)
for en in tqdm(all_epnames):
    ep_info = {}
    ep = episode_from_epname(en)
    #with open('SummScreen/transcripts/{en}.json') as f:
        #transcript_lines = json.load(f)['Transcript']
    if '[SCENE_BREAK]' in ep.transcript:
        ep_info['scene_breaks'] = 'explicit'
    elif '' in ep.transcript or any('--------' in x for x in ep.transcript):
        ep_info['scene_breaks'] = 'implicit'
    else:
        ep_info['scene_breaks'] = 'none'
    with open(f'SummScreen/closed_captions/{en}.json') as f:
        cc = json.load(f)
    ep_info['has_caps'] = 'captions' in cc.keys()
    with open(f'SummScreen/summaries/{en}.json') as f:
        summ = json.load(f)
    ep_info['has_summ'] = len(summ.keys())>0
    ep_info.update({sn:False for sn in ['soap_central', 'soapcentral_condensed', 'tvmega_recap', 'tvdb', 'has_summ']})
    for k,v in summ.items():
        if '[ RECAP AVAILABLE ]' not in v and 'Episode summary coming soon.' not in v and k not in ('imdb', 'yt', 'tvmega_summary'):
            if k=='imdb':
                print(v)
            ep_info['has_summ'] = True
            ep_info[k] = True
    #if ep_info['scene_breaks'] == 'none' and ep_info['has_caps']:
        #print(f'{en} has caps and no scene breaks')
    ep_info['n_scenes'] = len(ep.scenes)
    speaker_lines = [x for x in ep.transcript if ':' in x]
    ep_info['n_lines'] = len(speaker_lines)
    ep_info['n_chars'] = len(set(x.split(':')[0] for x in speaker_lines))
    n_chars_in_each_scene = [len(set(x.split(':')[0] for x in sc)) for sc in ep.scenes]
    ep_info['n_chars_per_scene'] = sum(n_chars_in_each_scene)/len(ep.scenes)
    #ep_info['duration_raw'] = cur_df.loc[en, 'duration_raw']
    #ep_info['duration'] = cur_df.loc[en, 'duration']
    try:
        ep_info['duration_raw'] = ffmpeg.probe(f'SummScreen/videos/{en}.mp4')['format']['duration']
        ep_info['duration'] = asMinutes(float(ep_info['duration_raw']))
    except ffmpeg._run.Error:
        print(f'failed to read vid for {en}')
        ep_info['duration_raw'] = 'failed video read'
        ep_info['duration'] = 'failed video read'
    dset_stats[en] = ep_info

df = pd.DataFrame(dset_stats).T
splits = ['val','test']*500 + ['train']*(len(df)-1000)
df['split'] = sorted(splits, key=lambda x:np.random.rand())
df['usable'] = df['has_summ'] & df['has_caps'] & (df['duration']!='failed video read')
df.to_csv('dset_info.csv')
