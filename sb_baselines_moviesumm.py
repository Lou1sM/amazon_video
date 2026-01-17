import os
import numpy as np
import pandas as pd
from dl_utils.misc import check_dir
from dl_utils.label_funcs import accuracy as cluster_acc
from utils import get_moviesumm_testnames, get_moviesumm_splitpoints, split_points_to_labels, segmentation_metrics
from tqdm import tqdm
import argparse
from scene_break_baselines import get_maybe_cached_psd_breaks, get_maybe_cached_yeo_labels, get_feats_and_times


parser = argparse.ArgumentParser()
parser.add_argument('--thresh', type=float, default=27)
parser.add_argument('--ndps', type=int, default=9999)
parser.add_argument('--method', type=str, default='psd')
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--dset', '-d', type=str, default='moviesumm')
ARGS = parser.parse_args()

all_method_names = ['psd-27', 'kmeans', 'GMM', 'berhe21', 'yeo96']#, 'psd-54']
test_vidnames, clean2cl = get_moviesumm_testnames()
all_scores = []
for vn in (pbar:=tqdm(test_vidnames[:ARGS.ndps])):
    if vn=='mr-turner_2014':
        continue
    kf_timepoints = np.load(f'data/ffmpeg-keyframes/{vn}/frametimes.npy')
    gt_split_points, betweens = get_moviesumm_splitpoints(vn)
    if len(gt_split_points)==0:
        print('No GT split points for', vn)
        continue
    kf_timepoints = kf_timepoints[~betweens]
    if ARGS.method == 'psd':
        vid_fp = f'data/full-videos/moviesumm/{vn}.mp4'
        pred_split_points = get_maybe_cached_psd_breaks(vn, ARGS.thresh, vid_fp, ARGS.recompute)
        pred_labels = split_points_to_labels(pred_split_points, kf_timepoints)
    elif ARGS.method == 'yeo96':
        feats_ar, ts = get_feats_and_times(vn)
        pred_labels = get_maybe_cached_yeo_labels(vn, feats_ar, delta=20, T=300)
        pred_labels = pred_labels[~betweens]
    elif ARGS.method == 'berhe21':
        feats_ar, ts = get_feats_and_times(vn)
        pred_labels = berhe21_preds(feats_ar, n_clusters=len(gt_split_points))
    gt_labels = split_points_to_labels(gt_split_points, kf_timepoints)
    scores = segmentation_metrics(pred_labels, gt_labels, k=30)
    scores['rev-acc'] = cluster_acc(gt_labels, pred_labels)
    all_scores.append(scores)
    print(vn, scores)
    #for i, scene in enumerate(scene_list):
    #    print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
    #        i+1,
    #        scene[0].get_timecode(), scene[0].get_frames(),
    #        scene[1].get_timecode(), scene[1].get_frames(),))

results_df = pd.DataFrame(all_scores)
results_df.loc['mean'] = results_df.mean(axis=0)
method_name_to_save = 'yeo96' if ARGS.method=='yeo96' else f'psd{ARGS.thresh}'
breakpoint()
results_df.to_csv(f'{method_name_to_save}-results.csv')
print(results_df)
breakpoint()
