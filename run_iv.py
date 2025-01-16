#import cv2
from time import time
import torch
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from natsort import natsorted
import json
import shutil
import re

from InternVideo.InternVideo2.multi_modality.demo_config import Config, eval_dict_leaf
from InternVideo.InternVideo2.multi_modality.demo.iv_utils import frames2tensor, setup_internvideo2

import argparse

show_name_dict = {
                  'friends':'Friends',
                  'house': 'House M.D.',
                  'met': 'How I Met Your Mother',
                  'bbt': 'The Big Bang Theory',
                  'castle': 'Castle',
                  'grey': "Grey's Anatomy",
                  }

with open('tvqa-splits.json') as f:
    tvqa_splits = json.load(f)

def extract_feats(show_name, season, ep):
    starttime = time()
    vid_subpath = f'tvqa/{show_name}/season_{season}/episode_{ep}'
    video_fp = f'data/full-videos/{vid_subpath}.mp4'

    #season_qs = full_dset_qs[show_name[0].upper()+show_name[1:]][f'season_{season}']
    season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{season}']
    try:
        dset_qs = season_qs[f'episode_{ep}']
    except KeyError:
        print(f'episode_{ep} not in season_qs')
        return
    scene_num = 0
    scene_frames = []
    scene_split_points = tvqa_splits[show_name][f'season_{season}'][f'episode_{ep}'][ARGS.splits]
    scene_split_points = np.array(scene_split_points)

    ep_id = f'{show_name}_s{season:02}e{ep:02}'
    with open(f'data/tvqa-transcripts/{ep_id}.json') as f:
        tlines_with_breaks = json.load(f)['Transcript']

    startsendsuts = []
    clip_start_time = 0
    cur_scene = []
    scenes = []
    cur_scene_idx = 0
    names_in_scenes = []
    if (not os.path.exists(text_cache_dir:=f'rag-caches/{vid_subpath}/{ARGS.splits}')) or ARGS.recompute_text_feats:
        for dir_name in ('text_feats', 'names', 'scene_texts'):
            fp=f'{text_cache_dir}/{dir_name}'
            #if os.path.exists(f'{text_cache_dir}/{dir_name}'):
                #print(f'{fp} already exists, removing')
                #shutil.rmtree(fp)
            #print(f'creating dir {fp}')
            os.makedirs(fp, exist_ok=True)
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
                with open(f'{text_cache_dir}/scene_texts/scene{len(scenes)}.txt', 'w') as f:
                    f.write('\n'.join(cur_scene))
                names =  list(set([x.split(' : ')[0].replace('Anabelle', 'Annabelle') for x in cur_scene if not x.startswith('UNK')]))
                with open(f'{text_cache_dir}/names/scene{len(scenes)}.txt', 'w') as f:
                    f.write('\n'.join(names))
                if cur_scene==[]:
                    stf = torch.zeros(1, 512)
                else:
                    stf = intern_model.get_txt_feat(cur_scene)
                print(f'writting text feats to {text_cache_dir}/text_feats/scene{cur_scene_idx}.pt')
                torch.save(stf, f'{text_cache_dir}/text_feats/scene{cur_scene_idx}.pt')
                #stf = text_model.get_txt_feat('\n'.join(cur_scene)).squeeze(0)
                scenes.append(cur_scene)
                names_in_scenes.append(names)
                cur_scene = []
                cur_scene_idx += 1
            cur_scene.append(ut_text)

    if (not os.path.exists(q_cache_dir:=f'rag-caches/{vid_subpath}/qfeats/{ARGS.splits}')) or ARGS.recompute_q_feats:
        os.makedirs(q_cache_dir, exist_ok=True)
        for i,qdict in enumerate(dset_qs['questions']):
            question = ' '.join(qdict[k] for k in ('q', 'a0', 'a1', 'a2', 'a3'))
            qvec = intern_model.get_txt_feat(question)
            torch.save(qvec, f'{q_cache_dir}/{i}.pt')


    if (not os.path.exists(vid_cache_dir:=f'data/internvid-feats/{vid_subpath}/{ARGS.splits}')) or ARGS.recompute_vid_feats:
        os.makedirs(vid_cache_dir, exist_ok=True)
        print(f'extracting vid feats to {vid_cache_dir}')
        vid = VideoFileClip(video_fp)
        vid_frames = list(vid.iter_frames())
        frame_splitpoints = (scene_split_points * vid.fps).astype(int)
        for i, frame in enumerate(vid_frames):
            if i in frame_splitpoints:
                if len(scene_frames)==0:
                    feats = torch.zeros(1,512)
                else:
                    frames_tensor = frames2tensor(scene_frames, fnum=4, target_size=(224, 224), device=ARGS.device)
                    feats = intern_model.get_vid_feat(frames_tensor)
                torch.save(feats, f'{vid_cache_dir}/scene{scene_num}.pt')
                scene_frames = []
                scene_num += 1
            else:
                scene_frames.append(frame)
    print(f'Extraction time for episode {ep}: {time()-starttime:.2f}s')

parser = argparse.ArgumentParser()
parser.add_argument('--show-name', type=str, default='friends')
parser.add_argument('--season', type=int, default='2')
parser.add_argument('--ep', type=int, default='2')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif'])
parser.add_argument('--recompute-text-feats', action='store_true')
parser.add_argument('--recompute-vid-feats', action='store_true')
parser.add_argument('--recompute-q-feats', action='store_true')
parser.add_argument('--recompute-all', action='store_true')
ARGS = parser.parse_args()

if ARGS.recompute_all:
    ARGS.recompute_text_feats = True
    ARGS.recompute_vid_feats = True
    ARGS.recompute_q_feats = True

with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

with open('tvqa_preprocessed_subtitles.json') as f:
    full_dset_subs = json.load(f)

config = Config.from_file('InternVideo/InternVideo2/multi_modality/demo/internvideo2_stage2_config.py')
config = eval_dict_leaf(config)
config.device = ARGS.device
config.model.text_encoder.config = 'InternVideo/InternVideo2/multi_modality/' + config.model.text_encoder.config
intern_model, tokenizer = setup_internvideo2(config)
#intern_model = intern_model.to(ARGS.device)
seasepshows = []
if ARGS.show_name=='all':
    show_names_to_compute = natsorted(os.listdir(f'data/full-videos/tvqa/'))
else:
    show_names_to_compute = [ARGS.show_name]
for show_name in show_names_to_compute:
    if ARGS.season == -1:
        seass_to_compute = natsorted([int(fn[7:]) for fn in os.listdir(f'data/full-videos/tvqa/{show_name}')])
    else:
        seass_to_compute = [ARGS.season]

    for seas in seass_to_compute:
        if ARGS.ep == -1:
            for fn in natsorted(os.listdir(f'data/full-videos/tvqa/{show_name}/season_{seas}')):
                ep_num = int(fn[8:].removesuffix('.mp4'))
                seasepshows.append((show_name, seas, ep_num))
        else:
            seasepshows.append((show_name, seas, ARGS.ep))

print(seasepshows)
for show_name, seas, ep in seasepshows:
    #try:
    extract_feats(show_name, seas, ep)
    #except Exception as e:
        #print(f'failed for season{seas} episode{ep}')
        #print(e)
