import os
import shutil
import re
from os.path import join
import numpy as np
import pandas as pd
import torch
import argparse
import json
from natsort import natsorted
from nltk.corpus import names
male_names = names.words('male.txt')
female_names = names.words('female.txt')
all_names = [n for n in male_names + female_names]


with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

with open('tvqa_preprocessed_subtitles.json') as f:
    full_dset_subs = json.load(f)

parser = argparse.ArgumentParser()
#parser.add_argument('--dset', '-d', type=str, default='tvqa')
parser.add_argument('--show-name', type=str, default='friends')
parser.add_argument('--season', type=int, default=2)
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--feats', type=str, required=True, choices=['internvideo', 'clip'])
parser.add_argument('--recompute-scene-texts', action='store_true')
ARGS = parser.parse_args()

ep_id = f'{ARGS.show_name}_s{ARGS.season:02}e{ARGS.episode:02}'
dset_name = 'tvqa'
vid_subpath = f'{dset_name}/{ARGS.show_name}/season_{ARGS.season}/episode_{ARGS.episode}'
kf_times = np.load(f'data/ffmpeg-keyframes/{vid_subpath}/frametimes.npy')
if ARGS.feats=='clip':
    import open_clip
    framefeatsdir = f'data/ffmpeg-frame-features/{vid_subpath}'
    clip_feats = np.stack([np.load(join(framefeatsdir, fn)) for fn in os.listdir(framefeatsdir)])
    split_idxs = np.load(f'data/ffmpeg-keyframes-by-scene/{vid_subpath}/scenesplit_idxs.npy')
    split_idxs = [0] + list(split_idxs) + [len(clip_feats)]
    scene_vision_feats = np.stack([clip_feats[split_idxs[i]:split_idxs[i+1]].mean(axis=0) for i in range(len(split_idxs)-1)])
    scene_vision_feats = torch.from_numpy(scene_vision_feats)

    model_name = 'hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K'
    text_model, preprocess = open_clip.create_model_from_pretrained(model_name)
    text_model.eval(); text_model.cuda()
    tokenizer = open_clip.get_tokenizer(model_name)
    #with torch.no_grad():
        #scene_text_feats = torch.stack([text_model.encode_text(tokenizer(s).cuda()).mean(axis=0) for s in scenes]).cpu()
    text_model = text_model.cpu()
else:
    vid_feats_dir = f'data/internvid-feats/tvqa/{ARGS.show_name}/season_{ARGS.season}/episode_{ARGS.episode}'
    scene_vision_feats = torch.stack([torch.load(join(vid_feats_dir, fn)) for fn in natsorted(os.listdir(vid_feats_dir))])

    from nltk.tokenize import word_tokenize
    from InternVideo.InternVideo2.multi_modality.demo_config import (Config,
                        eval_dict_leaf)

    from InternVideo.InternVideo2.multi_modality.demo.utils import setup_internvideo2

    config = Config.from_file('InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
    config = eval_dict_leaf(config)
    config.model.text_encoder.config = 'InternVideo/InternVideo2/multi_modality/' + config.model.text_encoder.config
    text_model, tokenizer = setup_internvideo2(config)

dset_qs = full_dset_qs[ARGS.show_name[0].upper()+ARGS.show_name[1:]][f'season_{ARGS.season}'][f'episode_{ARGS.episode}']
dset_subs = [x for x in full_dset_subs if x['vid_name'].startswith(f'{ARGS.show_name}_s{int(ARGS.season):02}e{int(ARGS.episode):02}')]

scene_split_points = np.load(f'data/ffmpeg-keyframes-by-scene/{vid_subpath}/scenesplit_timepoints.npy')
scene_split_points = np.array([0] + list(scene_split_points) + [float(kf_times.max())])

with open(f'data/tvqa-transcripts/{ep_id}.json') as f:
    tlines_with_breaks = json.load(f)['Transcript']

#tlines = [x for x in tlines_with_breaks if x!='[SCENE_BREAK]']
startsendsuts = []
clip_start_time = 0
cur_scene = []
scenes = []
cur_scene_idx = 0
names_in_scenes = []
scene_text_feats = []
if os.path.exists(scene_cache_dir:=f'rag-caches/{vid_subpath}') and not ARGS.recompute_scene_texts:
    scene_text_feats = [torch.load(f'{scene_cache_dir}/feats/{fn}') for fn in os.listdir(f'{scene_cache_dir}/feats')]
    names_in_scenes = []
    for fn in os.listdir(f'{scene_cache_dir}/names'):
        with open(f'{scene_cache_dir}/names/{fn}') as f:
            names_in_scenes.append(f.read().split('\n'))
    for fn in os.listdir(f'{scene_cache_dir}/scene_texts'):
        with open(f'{scene_cache_dir}/scene_texts/{fn}') as f:
            scenes.append(f.read().split('\n'))
else:
    for dir_name in ('feats', 'names', 'scene_texts'):
        shutil.rmtree(fp:=f'{scene_cache_dir}/{dir_name}')
        os.makedirs(fp)
    for i, tl in enumerate(tlines_with_breaks):
        if tl == '[SCENE_BREAK]':
            clip_start_time += end
            continue
        reobj = re.search(r'(.*) T: (.*)$', tl)
        start, end = reobj.group(2).split(' - ')
        start, end = float(start), float(end)
        ut_text = reobj.group(1).strip()
        startsendsuts.append([start+clip_start_time, end+clip_start_time, ut_text])

        if start+clip_start_time > scene_split_points[cur_scene_idx+1] or i==len(tlines_with_breaks)-1:
            if i==len(tlines_with_breaks)-1:
                cur_scene.append(ut_text)
            with open(f'{scene_cache_dir}/scene_texts/scene{len(scenes)}.txt', 'w') as f:
                f.write('\n'.join(cur_scene))
            names =  list(set([x.split(' : ')[0].replace('Anabelle', 'Annabelle') for x in cur_scene if not x.startswith('UNK')]))
            print(names)
            if any(n.startswith('An') for n in names):
                breakpoint()
            with open(f'{scene_cache_dir}/names/scene{len(scenes)}.txt', 'w') as f:
                f.write('\n'.join(names))
            stf = text_model.get_txt_feat(cur_scene)
            torch.save(stf, f'{scene_cache_dir}/feats/scene{i}.pt')
            #stf = text_model.get_txt_feat('\n'.join(cur_scene)).squeeze(0)
            scenes.append(cur_scene)
            scene_text_feats.append(stf)
            names_in_scenes.append(names)
            cur_scene = []
            cur_scene_idx += 1
        cur_scene.append(ut_text)

#stf = text_model.get_txt_feat(cur_scene).mean(axis=0)
#stf = text_model.get_txt_feat('\n'.join(cur_scene)).squeeze(0)
#scene_text_feats.append(stf)
#names_in_scenes.append(names)
uts_df = pd.DataFrame(startsendsuts, columns=['start', 'end', 'ut'])
assert uts_df['start'].tolist() == sorted(uts_df['start'].tolist())
assert uts_df['end'].tolist() == sorted(uts_df['end'].tolist())
clip_timepoints = np.cumsum([0] + [max(x['end'] for x in clip['sub']) for clip in sorted(dset_subs, key=lambda x:x['vid_name'])])

#scene_text_feats = torch.stack(scene_text_feats)
#scene_feats = (scene_vision_feats + scene_text_feats) / 2
scene_feats = scene_vision_feats


ious = []
n_correct = 0
for qdict in dset_qs['questions']:
    #qsent = tokenizer(qdict['q'])
    qvec = text_model.get_txt_feat(qdict['q'])
    vsims = (scene_vision_feats @ qvec.T).squeeze()
    tsims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_text_feats])
    sims = tsims + vsims
    names_in_q = [w.replace('Anabelle', 'Annabelle') for w in word_tokenize(qdict['q']) if w in all_names]
    names_match = torch.tensor([all(n in ns for n in names_in_q) for ns in names_in_scenes]).float()
    #sims = sims*names_match
    intersection = 0
    union = 0
    sims[~names_match.bool()] -= torch.inf

    gt_clip_idx = sorted(dset_qs['clips']).index(qdict['vid_name'])
    gt_start, gt_end = clip_timepoints[gt_clip_idx], clip_timepoints[gt_clip_idx+1]
    pred_scene_idxs = sims.topk(1).indices
    best_matching_scene = (scene_split_points<gt_start).argmin() - 1
    if not names_match[best_matching_scene]:
        breakpoint()
    if pred_scene_idxs[0]==best_matching_scene:
        n_correct += 1
    for psi in pred_scene_idxs:
        pred_start = scene_split_points[psi]
        pred_end = scene_split_points[psi+1]
        intersection += max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union += (max(pred_end, gt_end) - min(pred_start, gt_start))
        print(f'PRED: scene={psi}, start={pred_start:.2f}, end={pred_end:.2f}')
    pred_section = uts_df.loc[(uts_df['start']>pred_start) & (uts_df['start']<pred_end)]
    #iou = max(0, min(pred_end, gt_end) - max(pred_start, gt_start)) / (max(pred_end, gt_end) - min(pred_start, gt_start))
    #iou = max(0, min(pred_end, gt_end) - max(pred_start, gt_start)) / (pred_end - pred_start)
    iou = intersection / union
    ious.append(iou)
    print(f'GT: start={gt_start:.2f}, end={gt_end:.2f}, Best Matching Scene: {best_matching_scene}, sims={sims}')
    print(f'IOU={iou}')
print(f'mean iou: {np.array(ious).mean()}')
print(f'scene acc: {n_correct/len(dset_qs["questions"])}')

