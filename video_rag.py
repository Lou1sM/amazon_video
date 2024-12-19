import os
import re
from os.path import join
import numpy as np
import pandas as pd
import torch
import open_clip
import argparse
import json


with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

with open('tvqa_preprocessed_subtitles.json') as f:
    full_dset_subs = json.load(f)

parser = argparse.ArgumentParser()
#parser.add_argument('--dset', '-d', type=str, default='tvqa')
parser.add_argument('--show-name', type=str, required=True)
parser.add_argument('--season', type=int, required=True)
parser.add_argument('--episode', type=int, required=True)
ARGS = parser.parse_args()

ep_id = f'{ARGS.show_name}_s{ARGS.season:02}e{ARGS.episode:02}'
dset_name = 'tvqa'
vid_subpath = f'{dset_name}/{ARGS.show_name}/season_{ARGS.season}/episode_{ARGS.episode}'
kf_times = np.load(f'data/ffmpeg-keyframes/{vid_subpath}/frametimes.npy')
framefeatsdir = f'data/ffmpeg-frame-features/{vid_subpath}'
clip_feats = np.stack([np.load(join(framefeatsdir, fn)) for fn in os.listdir(framefeatsdir)])
dset_qs = full_dset_qs[ARGS.show_name[0].upper()+ARGS.show_name[1:]][f'season_{ARGS.season}'][f'episode_{ARGS.episode}']
dset_subs = [x for x in full_dset_subs if x['vid_name'].startswith(f'{ARGS.show_name}_s{int(ARGS.season):02}e{int(ARGS.episode):02}')]
split_idxs = np.load(f'data/ffmpeg-keyframes-by-scene/{vid_subpath}/scenesplit_idxs.npy')
split_idxs = [0] + list(split_idxs) + [len(clip_feats)]
scene_img_feats = np.stack([clip_feats[split_idxs[i]:split_idxs[i+1]].mean(axis=0) for i in range(len(split_idxs)-1)])
scene_img_feats = torch.from_numpy(scene_img_feats)

scene_split_points = np.load(f'data/ffmpeg-keyframes-by-scene/{vid_subpath}/scenesplit_timepoints.npy')
scene_split_points = [0] + list(scene_split_points) + [float(kf_times.max())]

with open(f'data/tvqa-transcripts/{ep_id}.json') as f:
    tlines_with_breaks = json.load(f)['Transcript']

#tlines = [x for x in tlines_with_breaks if x!='[SCENE_BREAK]']
startsendsuts = []
clip_start_time = 0
cur_scene = []
scenes = []
cur_scene_idx = 0
for i, tl in enumerate(tlines_with_breaks):
    if tl == '[SCENE_BREAK]':
        clip_start_time += end
        continue
    reobj = re.search(r'(.*) T: (.*)$', tl)
    start, end = reobj.group(2).split(' - ')
    start, end = float(start), float(end)
    startsendsuts.append([start+clip_start_time, end+clip_start_time, reobj.group(1)])

    if start+clip_start_time > scene_split_points[cur_scene_idx+1]:
        scenes.append(cur_scene)
        cur_scene = []
        cur_scene_idx += 1
    cur_scene.append(reobj.group(1))

scenes.append(cur_scene)
uts_df = pd.DataFrame(startsendsuts, columns=['start', 'end', 'ut'])
assert uts_df['start'].tolist() == sorted(uts_df['start'].tolist())
assert uts_df['end'].tolist() == sorted(uts_df['end'].tolist())
clip_timepoints = np.cumsum([0] + [max(x['end'] for x in clip['sub']) for clip in sorted(dset_subs, key=lambda x:x['vid_name'])])
model_name = 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
model, preprocess = open_clip.create_model_from_pretrained(model_name)
model.eval(); model.cuda()
tokenizer = open_clip.get_tokenizer(model_name)
with torch.no_grad():
    scene_text_feats = torch.stack([model.encode_text(tokenizer(s).cuda()).mean(axis=0) for s in scenes]).cpu()
model = model.cpu()
#scene_text_feats = model.encode_text(tokenizer(scenes))
assert scene_text_feats.shape == scene_img_feats.shape
#scene_feats = (scene_text_feats + scene_img_feats) / 2
scene_feats = scene_img_feats

for qdict in dset_qs['questions']:
    qsent = tokenizer(qdict['q'])
    qvec = model.encode_text(qsent)
    sims = scene_feats @ qvec.T
    pred_scene_idx = int(sims.argmin())
    pred_start = scene_split_points[pred_scene_idx]
    pred_end = scene_split_points[pred_scene_idx+1]
    gt_clip_idx = sorted(dset_qs['clips']).index(qdict['vid_name'])
    gt_start, gt_end = clip_timepoints[gt_clip_idx], clip_timepoints[gt_clip_idx+1]
    pred_section = uts_df.loc[(uts_df['start']>pred_start) & (uts_df['start']<pred_end)]
    print(f'PRED: scene={pred_scene_idx}, start={pred_start:.2f}, end={pred_end:.2f}')
    print(f'GT: start={gt_start:.2f}, end={gt_end:.2f}')
print([len(x) for x in scenes])

