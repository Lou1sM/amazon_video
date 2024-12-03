import os
from natsort import natsorted
from scene_detection import SceneSegmenter # ours
from scenedetect import detect, ContentDetector #pyscenedetect
import numpy as np
from sklearn.cluster import KMeans
import sys
import argparse
from osvd_names import vidname2fps as osvd_vn2fps
from get_bbc_times import annot_fns_list as bbc_annot_fns_list
from osvd_names import get_scene_split_times as osvd_scene_split_times
from get_bbc_times import get_scene_split_times as bbc_scene_split_times
from sklearn.mixture import GaussianMixture
from utils import segmentation_metrics, metric_names, bbc_mean_maxs
import pandas as pd
from dl_utils.misc import check_dir


all_method_names = ['kmeans', 'GMM', 'berhe21', 'yeo96']
def get_maybe_cached_psd_breaks(vn, thresh):
    check_dir(cachedir:=f'cached_outputs/pyscenedetect-cache/thresh{thresh}')
    if os.path.exists(cache_fp:=f'{cachedir}/{vn}.npy'):
        return np.load(cache_fp)
    scene_list = detect(f'data/moviesumm-videos/{vn}.mp4', ContentDetector(threshold=thresh))
    break_times = np.array([s[1].get_seconds() for s in scene_list[:-1]])
    print(break_times)
    np.save(cache_fp, break_times)
    return break_times

def get_maybe_cached_yeo_labels(vn, feats, delta, T):
    check_dir(cachedir:=f'cached_outputs/yeo96-cache/thresh{delta}-{T}')
    if os.path.exists(cache_fp:=f'{cachedir}/{vn}.npy'):
        return np.load(cache_fp)
    labels = yeo96_preds(feats, delta, T)
    np.save(cache_fp, labels)
    return labels

def get_baseline_preds(pred_name, feats, n_clusters):
    if pred_name=='kmeans':
        raw_clabels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(feats)
    elif pred_name=='GMM':
        raw_clabels = GaussianMixture(n_components=n_clusters).fit_predict(feats)
    elif pred_name=='berhe21':
        raw_clabels = berhe21_preds(feats, n_clusters)
    elif pred_name=='yeo96':
        raw_clabels = yeo96_preds(feats, ARGS.yeo_delta, T=ARGS.yeo_T)
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

def merge_at(orig, merged, i, j):
    start2i = list(orig[:i])
    i2j = list(orig[i+1:j])
    j2end = list(orig[j+1:])
    combined = start2i + [merged] + i2j + j2end
    assert len(combined) == len(orig)-1
    return combined

def yeo96_preds(feats, delta, T):
    cluster_means = feats
    N = len(feats)
    cluster_members = [[i] for i in range(N)]
    cluster_start_ends = np.array([(i,i) for i in range(N)])
    dists_ = (cluster_means - np.expand_dims(cluster_means,1))
    dists = np.linalg.norm(dists_, axis=2)
    while True:
        assert len(cluster_means) == len(cluster_start_ends)
        assert len(cluster_start_ends) == len(cluster_members)
        for i, (x_start, x_end) in enumerate(cluster_start_ends):
            for j, (y_start, y_end) in enumerate(cluster_start_ends):
                if abs(y_end - x_start)>T or abs(x_end - y_start)>T:
                    dists[i,j] = np.inf

        dists = dists + np.finfo(np.float32).max*(np.tri(len(cluster_means)))
        if dists.min() > delta or len(dists)==1:
            print(f'breaking because mindist is {dists.min():.3f} and delta is {delta}')
            break
        i,j = dists.argmin()//dists.shape[1], dists.argmin()%dists.shape[1]
        if not ( i<j):
            breakpoint()
        new_merged_mean = (cluster_means[i]+cluster_means[j])/2
        new_start_end = min(cluster_start_ends[i][0], cluster_start_ends[j][0]), max(cluster_start_ends[i][1], cluster_start_ends[j][1])
        new_merged_members = cluster_members[i]+cluster_members[j]
        new_dists = np.linalg.norm(cluster_means-new_merged_mean, axis=1)
        cluster_means = np.stack(merge_at(cluster_means, new_merged_mean, i, j))
        cluster_start_ends = np.stack(merge_at(cluster_start_ends, new_start_end, i, j))
        dists[i] = new_dists
        dists[:,i] = new_dists
        dists = np.delete(np.delete(dists, j, axis=0), j, axis=1)
        assert (dists[:j,i]==new_dists[:j]).all()
        cluster_members = merge_at(cluster_members, new_merged_members, i, j)
        #print(f'Merged {i} and {j}, cluster_means: {len(cluster_means)} cluster_start_ends: {len(cluster_start_ends)}  cluster_members: {len(cluster_members)}')

    cluster_labels = np.empty(N)
    for cl, members in enumerate(cluster_members):
        for m in members:
            cluster_labels[m] = cl

    print(f'Found {len(set(cluster_labels))} scenes for {len(cluster_labels)} dpoints')
    return np.array(cluster_labels)

def get_feats_and_times(vidname):
    framefeatsdir=f'data/ffmpeg-frame-features/{vidname}'
    ts = np.load(f'data/ffmpeg-keyframes/{vidname}/frametimes.npy')
    ts = [t for t in ts if t > ARGS.cut_first_n_secs]
    fns = [x for x in os.listdir(framefeatsdir) if x.endswith('.npy')]
    sorted_fns = natsorted(fns)
    feat_paths = [f'{framefeatsdir}/{x.split(".")[0]}.npy' for x in sorted_fns]
    feats_ar = np.array([np.load(featp) for featp in feat_paths])
    return feats_ar, ts

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, nargs='+', choices=all_method_names+['all'], required=True)
    parser.add_argument('--cut-first-n-secs', type=int, default=0)
    parser.add_argument('--yeo-delta', type=float, default=5.0)
    parser.add_argument('--yeo-T', type=int, default=300)
    parser.add_argument('-d', '--dset', type=str, default='osvd')
    ARGS = parser.parse_args()
    if ARGS.methods == ['all']:
        method_names = all_method_names
    else:
        method_names = ARGS.methods

    ss = SceneSegmenter(ARGS.dset)

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
        if ARGS.methods==['all']:
            check_dir('segmentation-results/osvd')
            results_df.to_csv('segmentation-results/osvd/baselines.csv')
        elif any(m.startswith('yeo96') for m in ARGS.methods):
            print(888)
            check_dir('segmentation-results/osvd')
            results_df.to_csv(f'segmentation-results/osvd/yeo96-{ARGS.yeo_delta}-{ARGS.yeo_T}.csv')
        print(results_df)

    if ARGS.dset=='bbc':
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
        max_avgs, mean_avgs = bbc_mean_maxs(df)
        if ARGS.methods==['all']:
            print(888)
            check_dir('segmentation-results/bbc')
            max_avgs.to_csv('segmentation-results/bbc-max/baselines.csv')
            mean_avgs.to_csv('segmentation-results/bbc-mean/baselines.csv')
        print('MAX')
        print(max_avgs)
        print('MEAN')
        print(mean_avgs)

