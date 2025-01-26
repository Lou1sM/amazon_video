import os
from natsort import natsorted
from scene_detection import SceneSegmenter
import numpy as np
import json
import argparse
from osvd_names import vidname2fps as osvd_vn2fps
from get_bbc_times import annot_fns_list as bbc_annot_fns_list
from osvd_names import get_scene_split_times as osvd_scene_split_times
from get_bbc_times import get_scene_split_times as bbc_scene_split_times
from utils import segmentation_metrics, metric_names
import pandas as pd
from dl_utils.misc import check_dir


parser = argparse.ArgumentParser()
parser.add_argument('--cut-first-n-secs', type=int, default=0)
parser.add_argument('-d', '--dset', type=str, default='osvd')
ARGS = parser.parse_args()

def convert_lgss_preds(raw_preds_fp):
    if raw_preds_fp.endswith('.json'):
        with open(raw_preds_fp) as f:
            raw_preds = json.load(f)
        pred_split_points = [float(v['frame'][0])/25 for v in raw_preds.values()]
    else:
        pred_split_points = np.load(raw_preds_fp)
    preds = (np.expand_dims(ts,1)>pred_split_points).sum(axis=1)
    return preds

if ARGS.dset=='osvd':
    avg_gt_scenes_dset = 22
    all_results = []
    for vn, fps in osvd_vn2fps.items():
        if isinstance(fps, str):
            continue
        ts = np.load(f'data/ffmpeg-keyframes/{vn}/frametimes.npy')
        ts = [t for t in ts if t > ARGS.cut_first_n_secs]
        ssts = osvd_scene_split_times(vn)
        preds = convert_lgss_preds(f'lgss-outputs/{vn}.npy')
        gt = [x[1] for x in ssts[:-1]]
        gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
        k = int(len(gt_point_labs)/(2*avg_gt_scenes_dset))
        results = segmentation_metrics(gt_point_labs, preds, k=k)
        all_results.append(results)
    results_df = pd.DataFrame(all_results).mean(axis=0)
    runtimes = np.load('lgss-outputs/osvd_runtimes.npy')
    nframes = np.load('data/osvd-nframes.npy')
    results_df['runtime'] = runtimes.mean()
    results_df['per-frame-runtime'] = (runtimes/nframes).mean()
    print(results_df)
    check_dir('segmentation-results/osvd')
    results_df.to_csv('segmentation-results/osvd/lgss.csv')

elif ARGS.dset=='bbc':
    avg_gt_scenes_dset = 48
    all_max_results = []
    all_mean_results = []
    fps = 25
    for vn in range(11):
        annotwise_ssts = bbc_scene_split_times(vn)
        ts = np.load(f'data/ffmpeg-keyframes/bbc_{vn+1:02}/frametimes.npy')
        ts = [t for t in ts if t > ARGS.cut_first_n_secs]
        k = int(len(ts)/(2*avg_gt_scenes_dset))
        preds = convert_lgss_preds(f'lgss-outputs/bbc_{vn+1:02}.npy')
        results = []
        for annot_num, gt in enumerate(annotwise_ssts):
            gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
            annot_results = segmentation_metrics(preds, gt_point_labs, k=k)
            results.append(annot_results)
        new_results_df = pd.DataFrame(results)
        all_mean_results.append(new_results_df.mean(axis=0))
        new_results_df['winddiff'] = - new_results_df['winddiff']
        new_results_df['ded'] = - new_results_df['ded']
        new_max_results = new_results_df.max(axis=0)
        new_max_results['winddiff'] = - new_max_results['winddiff']
        new_max_results['ded'] = - new_max_results['ded']
        all_max_results.append(new_max_results)

    runtimes = np.load('lgss-outputs/bbc_runtimes.npy')
    nframes = np.load('data/bbc-nframes.npy')
    max_avgs = pd.DataFrame(all_max_results).mean(axis=0)
    mean_avgs = pd.DataFrame(all_mean_results).mean(axis=0)
    max_avgs['runtime'] = runtimes.mean(); mean_avgs['runtime'] = runtimes.mean()
    max_avgs['per-frame-runtime'] = (runtimes/nframes).mean(); mean_avgs['per-frame-runtime'] = (runtimes/nframes).mean()
    os.makedirs('segmentation-results/bbc-max', exist_ok=True)
    os.makedirs('segmentation-results/bbc-mean', exist_ok=True)
    max_avgs.to_csv('segmentation-results/bbc-max/lgss.csv')
    mean_avgs.to_csv('segmentation-results/bbc-mean/lgss.csv')
    print('MAX')
    print(max_avgs)
    print('MEAN')
    print(mean_avgs)

