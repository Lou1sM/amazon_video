import os
from PIL import Image
from tqdm import tqdm
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
from sklearn.decomposition import PCA
from utils import segmentation_metrics, metric_names, bbc_mean_maxs, get_moviesumm_testnames, get_moviesumm_splitpoints, split_points_to_labels
import pandas as pd
from dl_utils.misc import check_dir
from time import time
import cv2


all_method_names = ['psd-27', 'kmeans', 'GMM', 'berhe21', 'yeo96', 'bassl', 'scrl', 'hem', 'hem-oracle', 'nn', 'stile']
def get_maybe_cached_psd_breaks(vn, thresh, vid_fp, recompute):
    check_dir(cachedir:=f'cached_outputs/pyscenedetect-cache/thresh{thresh}')
    if os.path.exists(cache_fp:=f'{cachedir}/{vn}.npy') and os.path.exists(runtime_cache_fp:=f'{cachedir}/{vn}-runtime') and not recompute:
        print(f'retrieving cache at {cache_fp}')
        with open(runtime_cache_fp) as f:
            runtime = float(f.read())
        break_times = np.load(cache_fp)
    else:
        starttime = time()
        #scene_list = detect(f'{vid_dir}/{vn}.mp4', ContentDetector(threshold=thresh))
        scene_list = detect(vid_fp, ContentDetector(threshold=thresh))

        break_times = np.array([s[1].get_seconds() for s in scene_list[:-1]])
        runtime = time()-starttime
        with open(f'{cachedir}/{vn}-runtime', 'w') as f:
            f.write(str(runtime))
        np.save(cache_fp, break_times)
    return break_times, runtime

def get_maybe_cached_yeo_labels(vn, feats, delta, T, recompute):
    check_dir(cachedir:=f'cached_outputs/yeo96-cache/thresh{delta}-{T}')
    if os.path.exists(cache_fp:=f'{cachedir}/{vn}.npy') and os.path.exists(runtime_cache_fp:=f'{cachedir}/{vn}-runtime') and len(np.load(cache_fp))==len(feats) and not recompute:
        print(f'retrieving cache at {cache_fp}')
        with open(runtime_cache_fp) as f:
            runtime = float(f.read())
        labels = np.load(cache_fp)
    else:
        starttime = time()
        labels = yeo96_preds(feats, delta, T)
        runtime = time()-starttime
        with open(f'{cachedir}/{vn}-runtime', 'w') as f:
            f.write(str(runtime))
        np.save(cache_fp, labels)
    return labels, runtime

def get_baseline_preds(pred_name, feats, n_clusters, vid_fp, recompute):
    feats_reduced = PCA(n_components=2).fit_transform(feats)
    starttime = time()
    vn = os.path.basename(vid_fp).removesuffix('.mp4')
    dset = os.path.basename(os.path.dirname(vid_fp))
    if pred_name.startswith('psd'):
        thresh = float(pred_name.split('-')[1])
        preds, runtime = get_maybe_cached_psd_breaks(vn, thresh, vid_fp=vid_fp, recompute=recompute)
    elif pred_name=='kmeans':
        starttime = time()
        raw_clabels = KMeans(n_clusters=n_clusters, n_init="auto").fit_predict(feats_reduced)
        runtime = time()-starttime
    elif pred_name=='GMM':
        starttime = time()
        raw_clabels = GaussianMixture(n_components=n_clusters, reg_covar=1e-5).fit_predict(feats_reduced)
        runtime = time()-starttime
    elif pred_name=='berhe21':
        starttime = time()
        raw_clabels = berhe21_preds(feats, n_clusters)
        runtime = time()-starttime
    elif pred_name=='yeo96':
        raw_clabels, runtime = get_maybe_cached_yeo_labels(vn, feats_reduced, ARGS.yeo_delta, T=ARGS.yeo_T, recompute=recompute)
    elif pred_name=='scrl':
        with open(f'../SceneSegmentation-SCRL/runtimes/{dset}/{vn}') as f:
            runtime = float(f.read())
        raw_clabels = np.load(f'../SceneSegmentation-SCRL/preds/{dset}/{vn}.npy')
    elif pred_name=='hem':
        starttime = time()
        silly_pooled = []
        for fn in os.listdir(f'data/ffmpeg-keyframes/{dset}/{vn}'):
            if fn.endswith('.jpg'):
                new = np.array(Image.open(f'data/ffmpeg-keyframes/{dset}/{vn}/{fn}'))
                sp = new.mean(axis=(0,1))
                silly_pooled.append(sp)
        changes = np.array([x@silly_pooled[i+1] for i,x in enumerate(silly_pooled[:-1])])
        split_idxs = np.argsort(changes)[:n_clusters-1]
        ts = np.load(f'data/ffmpeg-keyframes/{dset}/{vn}/frametimes.npy')
        split_points = ts[split_idxs]
        raw_clabels = split_points_to_labels(split_points, ts)
        runtime = time()-starttime
    elif pred_name=='bassl':
        with open(f'../bassl/runtimes/{dset}/{vn}') as f:
            runtime = float(f.read())
        raw_clabels = np.load(f'../bassl/results/{dset}/{vn}.npy')
    elif pred_name=='nn':
        with open(f'../NeighborNet/runtimes/{dset}/{vn}') as f:
            runtime = float(f.read())
        raw_clabels = np.load(f'../NeighborNet/results/{dset}/{vn}.npy')
    elif pred_name=='stile':
        with open(f'scene_tiling/runtimes/{dset}/{vn}') as f:
            runtime = float(f.read())
        raw_clabels = np.load(f'scene_tiling/results/{dset}/{vn}.npy')
    else:
        breakpoint()
    if not pred_name.startswith('psd'):
        running = 0
        uni_dim_seg_labels = []
        for i, cl in enumerate(raw_clabels):
            if i != 0 and raw_clabels[i-1] != raw_clabels[i]:
                running += 1
            uni_dim_seg_labels.append(running)
        preds = np.array(uni_dim_seg_labels)

    return preds, runtime

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
    #while True:
    for attempt in range(10):
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

def get_feats_and_times(dset_name, vidname, feats_name):
    framefeatsdir=f'data/ffmpeg-frame-features/{dset_name}/{feats_name}/{vidname}'
    ts = np.load(f'data/ffmpeg-keyframes/{dset_name}/{vidname}/frametimes.npy')
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
    parser.add_argument('--yeo-delta', type=float, default=20)
    parser.add_argument('--yeo-T', type=int, default=300)
    parser.add_argument('-d', '--dset', type=str, default='osvd')
    parser.add_argument('--feats-name', type=str, default='clip')
    parser.add_argument('--recompute', action='store_true')
    ARGS = parser.parse_args()
    if ARGS.methods == ['all']:
        method_names = all_method_names
    else:
        method_names = ARGS.methods


    if ARGS.dset=='osvd':
        avg_gt_scenes_dset = 22
        all_results = {m:{} for m in method_names}
        nframes = np.load('data/osvd-nframes.npy')
        pbar = tqdm(osvd_vn2fps.items())
        breakpoint()
        for vn, fps in pbar:
            if isinstance(fps, str):
                continue
            #if vn=='tos': continue
            pbar.set_description(vn)
            vid_fp=f'data/full-videos/osvd/{vn}.mp4'
            feats_ar, ts = get_feats_and_times(ARGS.dset, vn, ARGS.feats_name)
            ssts = osvd_scene_split_times(vn)
            gt_split_points = [x[1] for x in ssts[:-1]]
            #gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
            gt_point_labs = split_points_to_labels(gt_split_points, ts)
            k = int(len(gt_point_labs)/(2*avg_gt_scenes_dset))
            method_preds = []
            for mn in method_names:
                if mn.endswith('-oracle'):
                    ncs_to_give = len(gt_split_points)+1
                    mn = mn.removesuffix('-oracle')
                else:
                    ncs_to_give = avg_gt_scenes_dset
                preds, runtime = get_baseline_preds(mn, feats_ar, ncs_to_give, vid_fp, recompute=ARGS.recompute)
                method_preds.append((preds, runtime))

            for pred_name, (preds, runtime) in zip(method_names, method_preds):
                if pred_name.startswith('psd'):
                    n_points = max(1000, len(preds))
                    #gt_to_use = (np.expand_dims(np.arange(n_points),1)>gt).sum(axis=1)
                    gt_to_use = split_points_to_labels(gt_split_points, np.arange(n_points))
                    #preds_to_use = (np.expand_dims(np.arange(n_points),1)>preds).sum(axis=1)
                    preds_to_use = split_points_to_labels(preds, np.arange(n_points))
                else:
                    gt_to_use = gt_point_labs
                    preds_to_use = preds
                    if gt_to_use.shape != preds_to_use.shape:
                        gt_to_use = split_points_to_labels(gt_split_points, np.arange(len(preds_to_use)))
                results = segmentation_metrics(preds_to_use, gt_to_use, k=k)
                results['runtime'] = runtime
                all_results[pred_name][vn] = results
        combined = []
        for m in method_names:
            df = pd.DataFrame(all_results[m])
            df.loc['per-frame-runtime'] = df.loc['runtime']/nframes
            combined.append(df.mean(axis=1))
        results_df = pd.DataFrame(combined, index=method_names)[metric_names]
        if ARGS.methods==['all']:
            check_dir('segmentation-results/osvd')
            results_df.to_csv('segmentation-results/osvd/baselines.csv')
        elif any(m.startswith('yeo96') for m in ARGS.methods):
            print(888)
            check_dir('segmentation-results/osvd')
            results_df.to_csv(f'segmentation-results/osvd/yeo96-{ARGS.yeo_delta}-{ARGS.yeo_T}.csv')
        print(results_df)
        print(' & '.join(f'{float(x)*100:.2f}' for x in results_df.iloc[0]))

    elif ARGS.dset=='bbc':
        avg_gt_scenes_dset = 48
        all_results_by_annot = [{m:{} for m in method_names} for _ in range(5)]
        nframes = np.load('data/bbc-nframes.npy')
        pbar = tqdm(range(11))
        for vn in pbar:
            basefn = f'bbc_{vn+1:02}'
            pbar.set_description(basefn)
            vid_fp=f'data/full-videos/bbc/{basefn}.mp4'
            assert os.path.exists(vid_fp)
            annotwise_ssts = bbc_scene_split_times(vn)
            feats_ar, ts = get_feats_and_times(ARGS.dset, basefn)
            k = int(len(ts)/(2*avg_gt_scenes_dset))
            method_preds = []
            runtimes = []
            for mn in method_names:
                if mn.endswith('-oracle'):
                    ncs_to_give = len(annotwise_ssts[0])+1
                    mn = mn.removesuffix('-oracle')
                else:
                    ncs_to_give = avg_gt_scenes_dset
                preds, rt = get_baseline_preds(mn, feats_ar, ncs_to_give, vid_fp=vid_fp, recompute=ARGS.recompute)
                method_preds.append(preds)
                runtimes.append(rt)

            for annot_num, gt in enumerate(annotwise_ssts):
                #gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
                gt_point_labs = split_points_to_labels(gt, ts)
                for i, (pred_name, preds) in enumerate(zip(method_names, method_preds)):
                    if pred_name.startswith('psd'):
                        n_points = max(1000, len(preds))
                        #gt_to_use = (np.expand_dims(np.arange(n_points),1)>gt).sum(axis=1)
                        gt_to_use = split_points_to_labels(gt, np.arange(n_points))
                        #preds_to_use = (np.expand_dims(np.arange(n_points),1)>preds).sum(axis=1)
                        preds_to_use = split_points_to_labels(preds, np.arange(n_points))
                    else:
                        gt_to_use = gt_point_labs
                        preds_to_use = preds
                        if gt_to_use.shape != preds_to_use.shape:
                            if gt_to_use.shape[0] != preds_to_use.shape[0]-1:
                                breakpoint()
                            preds_to_use = preds_to_use[:-1]
                            #gt_to_use = split_points_to_labels(gt_split_points, np.arange(len(preds_to_use)))
                    results = segmentation_metrics(gt_to_use, preds_to_use, k=k)
                    results['runtime'] = runtimes[i]
                    results['per-frame-runtime'] = runtimes[i] / nframes[i]
                    all_results_by_annot[annot_num][pred_name][vn] = results

        df=pd.json_normalize(all_results_by_annot)
        df.columns = pd.MultiIndex.from_tuples([tuple(col.split('.')) for col in df.columns])
        max_avgs, mean_avgs = bbc_mean_maxs(df)
        max_avgs = max_avgs[metric_names]; mean_avgs = mean_avgs[metric_names]
        print('MAX')
        print(max_avgs)
        print(' & '.join((f'{x:.2f}' for x in max_avgs.iloc[0].values*100)))
        print('MEAN')
        print(mean_avgs)
        print(' & '.join((f'{x:.2f}' for x in mean_avgs.iloc[0].values*100)))
        if ARGS.methods==['all']:
            print(888)
            check_dir('segmentation-results/bbc')
            max_avgs.to_csv('segmentation-results/bbc-max/baselines.csv')
            mean_avgs.to_csv('segmentation-results/bbc-mean/baselines.csv')

    elif ARGS.dset=='moviesumm':
        avg_gt_scenes_dset = 50
        all_results = {m:{} for m in method_names}
        test_vidnames, clean2cl = get_moviesumm_testnames()
        nframes = np.load('data/moviesumm-nframes.npy')
        pbar = tqdm(test_vidnames)
        for vn in pbar:
            if vn=='the-insider_1999': continue
            pbar.set_description(vn)
            vid_fp=f'data/full-videos/moviesumm/{vn}.mp4'
            gt_split_points, betweens = get_moviesumm_splitpoints(vn)
            if method_names==['psd-27']:
                feats_ar, ts = None, None
                k = 1
            else:
                try:
                    feats_ar, ts = get_feats_and_times(ARGS.dset, vn)
                    kf_timepoints = np.load(f'data/ffmpeg-keyframes/{ARGS.dset}/{vn}/frametimes.npy')
                except FileNotFoundError:
                    continue
                try:
                    kf_timepoints = kf_timepoints[~betweens]
                    feats_ar = feats_ar[~betweens]
                except IndexError as e:
                    print(f'error trying to remove betweens: {e} for {vn}, just leaving them in')

                gt_point_labs = split_points_to_labels(gt_split_points, kf_timepoints)

                k = int(len(gt_point_labs)/(2*avg_gt_scenes_dset))
            method_preds = []
            try:
                for mn in method_names:
                    preds, runtime = get_baseline_preds(mn, feats_ar, avg_gt_scenes_dset, vid_fp, recompute=ARGS.recompute)
                    method_preds.append((preds, runtime))
            except OSError as e:
                print(f'error {e} for {vn} {mn}')

            for pred_name, (preds, runtime) in zip(method_names, method_preds):
                if pred_name.startswith('psd'):
                    n_points = max(1000, len(preds))
                    gt_to_use = split_points_to_labels(gt_split_points, np.arange(n_points))
                    preds_to_use = split_points_to_labels(preds, np.arange(n_points))
                else:
                    gt_to_use = gt_point_labs
                    preds_to_use = preds
                    if gt_to_use.shape != preds_to_use.shape:
                        gt_to_use = split_points_to_labels(gt_split_points, np.arange(len(preds_to_use)))
                results = segmentation_metrics(preds_to_use, gt_to_use, k=k)
                results['runtime'] = runtime
                all_results[pred_name][vn] = results
        combined = []
        for m in method_names:
            df = pd.DataFrame(all_results[m])
            df.loc['per-frame-runtime'] = df.loc['runtime']#/nframes[:df.shape[1]]
            combined.append(df.mean(axis=1))
        results_df = pd.DataFrame(combined, index=method_names)[metric_names]
        if ARGS.methods==['all']:
            check_dir('segmentation-results/moviesumm')
            results_df.to_csv('segmentation-results/moviesumm/baselines.csv')
        elif any(m.startswith('yeo96') for m in ARGS.methods):
            print(888)
            check_dir('segmentation-results/moviesumm')
            results_df.to_csv(f'segmentation-results/moviesumm/yeo96-{ARGS.yeo_delta}-{ARGS.yeo_T}.csv')
        print(results_df)
