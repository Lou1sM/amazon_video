import json
import math
from nltk.tokenize import sent_tokenize
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from dl_utils.label_funcs import accuracy as cluster_acc
from dl_utils.misc import time_format
import pandas as pd
import numpy as np
#from utils import get_all_testnames
from datasets import load_dataset
from difflib import SequenceMatcher
from align_vid_and_transcripts import align
import re
from tqdm import tqdm
import argparse
from dl_utils.label_funcs import accuracy
from lingua import Language, LanguageDetectorBuilder


acc = lambda x,y: (cluster_acc(x,y) + cluster_acc(y,x))/2

def get_all_testnames():
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

parser = argparse.ArgumentParser()
parser.add_argument('--only-fix-transcripts', action='store_true')
parser.add_argument('--ndps', type=int, default=99999)
ARGS = parser.parse_args()

test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}

ds = load_dataset("rohitsaxena/MovieSum")

metric_names = ['acc', 'ari', 'nmi']
method_names = ['ours', 'un60', 'un75', 'un90', 'unorc']
results = {mt: {mc:[] for mc in metric_names} for mt in method_names}

all_accs = []
all_nmis = []
all_aris = []
test_vidnames = ['the-sixth-sense_1999'] + test_vidnames

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

n_pred_sceness = []
for vn in (pbar:=tqdm(test_vidnames[:ARGS.ndps])):
    if vn=='mr-turner_2014':
        continue
    #if vn!='somethings-gotta-give_2003':
        #continue
    with open(f'data/whisper_outputs/{vn}.json') as f:
       wlines = json.load(f)

    wlines = [ut for ut in wlines if not set(ut['text'].split())==set(['you'])]
    wlines = [ut for ut in wlines if detector.detect_language_of(ut['text']) == Language.ENGLISH]
    wspoken = [ut['text'] for ut in wlines]

    gt_match_name = cl2clean[vn]
    gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
    gt_script = gt_match['script']
    gt_spoken = []
    all_scene_idxs = []
    scene_idx = 0
    for l in gt_script.split('\n'):
        l = l.strip()
        if l.startswith('<scene>'):
            scene_idx += 1
        elif l.startswith('<dialogue>'):
            spoken = l.removeprefix('<dialogue>').removesuffix('</dialogue>')
            gt_spoken.append(spoken)
            all_scene_idxs.append(scene_idx)

    alignment = align(wspoken, gt_spoken)
    assert len(wlines)==len(wspoken)
    assert len(all_scene_idxs)==len(gt_spoken)
    #gt = np.array(all_scene_idxs)[alignment.index2]
    #prev_endtime = 0
    curr_starttime = None
    curr_endtime = 0
    prev_sidx = all_scene_idxs[0]
    all_startends = []
    for i,j in zip(alignment.index1, alignment.index2):
        wut = wlines[i]
        assert wut['end']>=curr_endtime
        if curr_starttime is None:
            curr_starttime = wut['start']
        else:
            assert wut['start']>=curr_starttime

        curr_endtime=wut['end']
        sidx = all_scene_idxs[j]
        if sidx!=prev_sidx:
            all_startends.append((curr_starttime,curr_endtime))
            prev_sidx = sidx
            curr_starttime = None

    all_starts = np.array([x[0] for x in all_startends])
    all_ends = np.array([x[1] for x in all_startends])
    gt_split_points = (all_starts[1:] + all_ends[:-1]) / 2
    pred_split_points = np.load(f'data/ffmpeg-keyframes-by-scene/{vn}/scene-split-timepoints.npy')
    kf_timepoints = np.load(f'data/ffmpeg-keyframes/{vn}/frametimes.npy')
    betweens = np.array([any(x>all_startends[i][1] and x<all_startends[i+1][0] for i in range(len(all_startends)-1)) for x in kf_timepoints])
    kf_timepoints = kf_timepoints[~betweens]
    #ts = np.arange(0,kf_timepoints[-1],0.1)
    ts = kf_timepoints
    gt_point_labs = (np.expand_dims(ts,1)>gt_split_points).sum(axis=1)
    pred_point_labs = (np.expand_dims(ts,1)>pred_split_points).sum(axis=1)
    gt_n_scenes = len(all_startends)
    if gt_n_scenes <= 1:
        continue
    unif_point_labs90 = np.linspace(0,90,len(ts)).astype(int)
    unif_point_labs75 = np.linspace(0,75,len(ts)).astype(int)
    unif_point_labs60 = np.linspace(0,60,len(ts)).astype(int)
    n_to_repeat = int(math.ceil(len(gt_point_labs)/gt_n_scenes))
    uniforc_point_labs = np.repeat(np.arange(gt_n_scenes), n_to_repeat)[:len(gt_point_labs)]
    n_pred_sceness.append(len(set(pred_point_labs)))
    #pred_point_labs = np.concatenate([pred_point_labs, np.array([pred_point_labs.max()+1]*n_betweens)])
    #gt_point_labs_ = np.concatenate([gt_point_labs, np.array([gt_point_labs.max()+1]*n_betweens)])
    #gt_point_labs_ = np.concatenate([gt_point_labs, np.array([gt_point_labs.max()+1]*n_betweens)])
    for pred_name, preds, in zip(method_names, [pred_point_labs, unif_point_labs60, unif_point_labs75, unif_point_labs90, uniforc_point_labs]):
        #gt_target = gt_point_labs_ if pred_name=='ours' else gt_point_labs
        gt_target = gt_point_labs
        for mname ,mfunc in zip(['acc','nmi','ari'], [acc, nmi, ari]):
            score = mfunc(gt_target, preds)
            results[pred_name][mname].append(score)

    pbar.set_description('  '.join(f'{k}: ' +
        ' '.join(f'{k1}: {np.array(v1).mean():0.2f}' for k1,v1 in v.items())
        for k,v in results.items()))
final_results = {k1:{k2:np.array(v2).mean() for k2,v2 in v1.items()} for k1,v1 in results.items()}
final_results = pd.DataFrame(final_results)
print(final_results)
final_results.to_csv('scenes-results.csv')

with open('scene-results.json.safe', 'w') as f:
    json.dump(results, f)
print(f'Mean num pred scenes: {np.array(n_pred_sceness).mean():.3f}')
