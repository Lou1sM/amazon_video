from tqdm import tqdm
from time import time
import torch
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from natsort import natsorted
import json
import shutil
import re
from utils import get_showseaseps
from line_profiler import LineProfiler

from InternVideo.InternVideo2.multi_modality.demo_config import Config, eval_dict_leaf
from InternVideo.InternVideo2.multi_modality.demo.iv_utils import frames2tensor, setup_internvideo2

import argparse

show_name_dict = {
                  'friends':'Friends',
                  'house': 'House M.D.',
                  'met': 'How I Met You Mother',
                  'bbt': 'The Big Bang Theory',
                  'castle': 'Castle',
                  'grey': "Grey's Anatomy",
                  }

def profile_aggregate(func):
    """
    Decorator that accumulates timing stats across multiple function calls
    and prints results only at the end of the program
    """
    # Create a single profiler instance that persists across calls
    profiler = LineProfiler()
    wrapped = profiler(func)

    def wrapper(*args, **kwargs):
        return wrapped(*args, **kwargs)

    # Store the profiler so we can access it later
    wrapper.profiler = profiler
    return wrapper

def profile_lines(func):
    """
    Decorator to profile specific lines within a function
    """
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler_wrapper = profiler(func)
        result = profiler_wrapper(*args, **kwargs)

        print("\n=== Line-by-line profiling ===")
        profiler.print_stats()

        return result
    return wrapper

def print_stats(func):
    """Call this after all executions to see accumulated stats"""
    if hasattr(func, 'profiler'):
        func.profiler.print_stats()
    else:
        print("No profiler found on function")

with open('../LLaVA-NeXT/tvqa-splits.json') as f:
    tvqa_splits = json.load(f)

mismatches = []

def make_maybe_clear(fp):
    if ARGS.clear_existing:
        if os.path.exists(fp):
            print(f'{fp} already exists, removing')
            shutil.rmtree(fp)
        print(f'creating dir {fp}')
    os.makedirs(fp, exist_ok=True)

#@profile_lines
def extract_feats(show_name, season, ep):
    global mismatches
    vid_subpath = f'tvqa/{show_name}/season_{season}/episode_{ep}'
    video_fp = f'data/full-videos/{vid_subpath}.mp4'

    #season_qs = full_dset_qs[show_name[0].upper()+show_name[1:]][f'season_{season}']
    season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{season}']
    try:
        dset_qs = season_qs[f'episode_{ep}']
    except KeyError:
        print(f'episode_{ep} not in season_qs')
        return
    scene_frames = []
    #if ARGS.splits in ['GMM', 'bassl']:
        #scene_split_points = np.load(f'data/baseline-ffmpeg-keyframes-by-scene/{ARGS.splits}/{vid_subpath}/scenesplit_timepoints.npy')
    #else:
    scene_split_points = tvqa_splits[ARGS.splits][show_name][f'season_{season}'][f'episode_{ep}']
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
    if ARGS.verbose:
        print(scene_split_points)
    if (not os.path.exists(cache_dir:=f'rag-caches/{ARGS.splits}/{vid_subpath}')) or ARGS.recompute_text_feats:
        for dir_name in ('text_feats', 'names', 'scene_texts'):
            fp=f'{cache_dir}/{dir_name}'
            make_maybe_clear(fp)
        writing_at = []
        for i, tl in enumerate(tlines_with_breaks):
            if tl == '[SCENE_BREAK]':
                clip_start_time += end
                continue
            reobj = re.search(r'(.*) T: (.*)$', tl)
            start, end = reobj.group(2).split(' - ')
            start, end = float(start), float(end)
            ut_text = reobj.group(1).strip()
            startsendsuts.append([start+clip_start_time, end+clip_start_time, ut_text])

            if (cur_scene_idx+1 < len(scene_split_points) and start+clip_start_time > scene_split_points[cur_scene_idx+1]) or i==len(tlines_with_breaks)-1:
                if ARGS.verbose:
                    print(start+clip_start_time)
                writing_at.append(start+clip_start_time)
                if i==len(tlines_with_breaks)-1:
                    cur_scene.append(ut_text)
                with open(f'{cache_dir}/scene_texts/scene{cur_scene_idx}.txt', 'w') as f:
                    f.write('\n'.join(cur_scene))
                names =  list(set([x.split(' : ')[0].replace('Anabelle', 'Annabelle') for x in cur_scene if not x.startswith('UNK')]))
                with open(f'{cache_dir}/names/scene{cur_scene_idx}.txt', 'w') as f:
                    f.write('\n'.join(names))
                if cur_scene==[]:
                    stf = torch.zeros(1, 512)
                elif cur_scene_idx%4==0:
                    with torch.no_grad():
                        stf = intern_model.get_txt_feat(cur_scene)
                torch.save(stf, f'{cache_dir}/text_feats/scene{cur_scene_idx}.pt')
                #stf = text_model.get_txt_feat('\n'.join(cur_scene)).squeeze(0)
                scenes.append(cur_scene)
                names_in_scenes.append(names)
                cur_scene = []
                cur_scene_idx += 1
                assert cur_scene_idx==len(scenes)
            cur_scene.append(ut_text)
        assert len(os.listdir(f'{cache_dir}/scene_texts'))==len(os.listdir(f'{cache_dir}/names'))
        assert len(os.listdir(f'{cache_dir}/scene_texts'))==len(os.listdir(f'{cache_dir}/text_feats'))
        while cur_scene_idx < len(scene_split_points)-1: # if ran out of tlines before surpassed penultimate split point
            torch.save(torch.zeros(1,512), f'{cache_dir}/text_feats/scene{cur_scene_idx}.pt')
            with open(f'{cache_dir}/scene_texts/scene{cur_scene_idx}.txt', 'w') as f:
                f.write('')
            with open(f'{cache_dir}/names/scene{cur_scene_idx}.txt', 'w') as f:
                f.write('')
            cur_scene_idx += 1
    if not len(os.listdir(f'{cache_dir}/text_feats/')) == len(scene_split_points)-1:
        mismatches.append((show_name, season, ep))
        print('text mismatch for',show_name, season, ep)
        return

    if (not os.path.exists(vid_cache_dir:=f'{cache_dir}/vid_feats/')) or ARGS.recompute_vid_feats:
        make_maybe_clear(vid_cache_dir)
        vid = VideoFileClip(video_fp)
        vid_frames = list(vid.iter_frames())
        frame_splitpoints = (scene_split_points * vid.fps).astype(int)
        if frame_splitpoints[-1] != len(vid_frames):
            print(f'mismmatched last frame_splitpoint={frame_splitpoints[-1]} and len(vid_frames)={len(vid_frames)}, fixing former to the latter')
            frame_splitpoints[-1] = len(vid_frames)
        writing_at = []
        writing_nums = []
        scene_num = 0
        minfnum = 10
        for i, frame in enumerate(vid_frames):
            if i+1 in frame_splitpoints:
                writing_at.append(i)
                writing_nums.append(scene_num)
                save_fp = f'{vid_cache_dir}/scene{scene_num}.pt'
                if scene_num%4 == 0:
                    if len(scene_frames)<minfnum:
                        feats = torch.zeros(1,512)
                    else:
                        frames_tensor = frames2tensor(scene_frames, fnum=4, target_size=(224, 224), device=ARGS.device)
                        print(f'enough frames at {len(scene_frames)}')
                        with torch.no_grad():
                            feats = intern_model.get_vid_feat(frames_tensor)
                    torch.save(feats, save_fp)
                else:
                    shutil.copy(f'{vid_cache_dir}/scene{scene_num-1}.pt', save_fp)
                scene_frames = []
                scene_num += 1
            else:
                scene_frames.append(frame)
        if not ( len(os.listdir(vid_cache_dir)) == len(scene_split_points)-1):
            mismatches.append((show_name, season, ep))
            print('vid mismatch for',show_name, season, ep)
            print('texts:', len(os.listdir(f'{cache_dir}/text_feats/')), 'split-points:', len(scene_split_points)-1)
            return

    print('n vids=', len(os.listdir(vid_cache_dir)), 'n texts=', len(os.listdir(f"{cache_dir}/text_feats/")), 'n names', len(os.listdir(f"{cache_dir}/names")), 'n text_feats', len(os.listdir(f"{cache_dir}/text_feats")))
    if (not os.path.exists(q_cache_dir:=f'rag-caches/qfeats/{vid_subpath}/')) or ARGS.recompute_q_feats:
        os.makedirs(q_cache_dir, exist_ok=True)
        for i,qdict in enumerate(dset_qs['questions']):
            question = ' '.join(qdict[k] for k in ('q', 'a0', 'a1', 'a2', 'a3'))
            with torch.no_grad():
                qvec = intern_model.get_txt_feat(question)
            torch.save(qvec, f'{q_cache_dir}/{i}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default='2')
    parser.add_argument('--ep', type=int, default='2')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif', 'GMM', 'scrl', 'bassl'])
    parser.add_argument('--recompute-text-feats', action='store_true')
    parser.add_argument('--recompute-vid-feats', action='store_true')
    parser.add_argument('--recompute-q-feats', action='store_true')
    parser.add_argument('--recompute-all', action='store_true')
    parser.add_argument('--clear-existing', action='store_true')
    parser.add_argument('--verbose', action='store_true')
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
    intern_model.half()
    #intern_model = intern_model.to(ARGS.device)
    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)
    print(showseaseps)
    for show_name, seas, ep in tqdm(showseaseps):
        #try:
        extract_feats(show_name, seas, ep)
        #except Exception as e:
            #print(f'failed for season{seas} episode{ep}')
            #print(e)
    print(mismatches)
    with open('mismatches.txt', 'w') as f:
        f.write('\n'.join(f'{m[0]} {m[1]} {m[2]}' for m in mismatches))
