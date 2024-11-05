import os
from natsort import natsorted
from scene_detection import SceneSegmenter
import numpy as np
from sklearn.cluster import KMeans
import sys
import argparse
from osvd_names import vidname2fps as osvd_vn2fps
from get_bbc_times import annot_fns_list as bbc_annot_fns_list
from osvd_names import get_scene_split_times as osvd_scene_split_times
from get_bbc_times import get_scene_split_times as bbc_scene_split_times
from sklearn.mixture import GaussianMixture
from utils import segmentation_metrics, metric_names
import pandas as pd
from dl_utils.misc import check_dir


all_method_names = ['kmeans', 'GMM', 'berhe21']
parser = argparse.ArgumentParser()
parser.add_argument('--methods', type=str, nargs='+', choices=all_method_names+['all'], required=True)
parser.add_argument('--cut-first-n-secs', type=int, default=0)
parser.add_argument('-d', '--dset', type=str, default='osvd')
ARGS = parser.parse_args()
if ARGS.methods == ['all']:
    method_names = all_method_names
else:
    method_names = ARGS.methods

ss = SceneSegmenter(ARGS.dset)

def get_baseline_preds(pred_name, feats, n_clusters):
    if pred_name=='kmeans':
        raw_clabels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(feats)
    elif pred_name=='GMM':
        raw_clabels = GaussianMixture(n_components=n_clusters).fit_predict(feats)
    elif pred_name=='berhe21':
        raw_clabels = berhe21_preds(feats, n_clusters)
    else:
        breakpoint()
    running = 0
    uni_dim_seg_labels = []
    for i, cl in enumerate(raw_clabels):
        if i != 0 and raw_clabels[i-1] != raw_clabels[i]:
            running += 1
        uni_dim_seg_labels.append(running)

    return np.array(uni_dim_seg_labels)

def berhe21_preds(feats, n_clusters):
    patience = 3; window_size = 3
    raw_clabels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(feats)
    running = 0
    uni_dim_seg_labels = []
    count = 0
    tmplist = list(raw_clabels[:window_size])
    for i, cl in enumerate(raw_clabels):
        if i < window_size or cl==tmplist[-1]:
            pass
        elif cl not in tmplist:
            count += 1
            if len(tmplist) < window_size:
                tmplist.append(cl)
        #else:
        tmplist = tmplist[1:] + [cl]
        if count==patience:
            running += 1
            count = 0
        uni_dim_seg_labels.append(running)

    return np.array(uni_dim_seg_labels)

def get_feats_and_times(vidname):
    framefeatsdir=f'data/ffmpeg-frame-features/{vidname}'
    ts = np.load(f'data/ffmpeg-keyframes/{vidname}/frametimes.npy')
    ts = [t for t in ts if t > ARGS.cut_first_n_secs]
    fns = [x for x in os.listdir(framefeatsdir) if x.endswith('.npy')]
    sorted_fns = natsorted(fns)
    feat_paths = [f'{framefeatsdir}/{x.split(".")[0]}.npy' for x in sorted_fns]
    feats_ar = np.array([np.load(featp) for featp in feat_paths])
    return feats_ar, ts

if ARGS.dset=='osvd':
    avg_gt_scenes_dset = 22
    all_results = {m:{} for m in method_names}
    for vn, fps in osvd_vn2fps.items():
        if isinstance(fps, str):
            continue
        feats_ar, ts = get_feats_and_times(vn)
        ssts = osvd_scene_split_times(vn)
        gt = [x[1] for x in ssts[:-1]]
        gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
        k = int(len(gt_point_labs)/(2*avg_gt_scenes_dset))
        method_preds = [get_baseline_preds(m, feats_ar, avg_gt_scenes_dset) for m in method_names]
        for pred_name, preds in zip(method_names, method_preds):
            results = segmentation_metrics(preds, gt_point_labs, k=k)
            all_results[pred_name][vn] = results
    combined = []
    for m in method_names:
        combined.append(pd.DataFrame(all_results[m]).mean(axis=1))
    results_df = pd.DataFrame(combined, index=method_names)[metric_names]
    print(results_df)
    check_dir('segmentation-results/osvd')
    results_df.to_csv('segmentation-results/osvd/baselines.csv')

elif ARGS.dset=='bbc':
    avg_gt_scenes_dset = 48
    all_results_by_annot = [{m:{} for m in method_names} for _ in range(5)]
    for vn in range(11):
        annotwise_ssts = bbc_scene_split_times(vn)
        feats_ar, ts = get_feats_and_times(f'bbc_{vn+1:02}')
        k = int(len(ts)/(2*avg_gt_scenes_dset))
        method_preds = [get_baseline_preds(m, feats_ar, avg_gt_scenes_dset) for m in method_names]
        for annot_num, gt in enumerate(annotwise_ssts):
            gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
            for pred_name, preds in zip(method_names, method_preds):
                results = {}
                results = segmentation_metrics(gt_point_labs, preds, k=k)
                all_results_by_annot[annot_num][pred_name][vn] = results

    df=pd.json_normalize(all_results_by_annot)
    df.columns = pd.MultiIndex.from_tuples([tuple(col.split('.')) for col in df.columns])
    max_avgs = df.max(axis=0).unstack().groupby(axis=0, level=0).mean()
    mean_avgs = df.mean(axis=0).unstack().groupby(axis=0, level=0).mean()
    check_dir('segmentation-results/bbc')
    max_avgs.to_csv('segmentation-results/bbc/baselines-max.csv')
    mean_avgs.to_csv('segmentation-results/bbc/baselines-mean.csv')
    print('MAX')
    print(max_avgs)
    print('MEAN')
    print(mean_avgs)

