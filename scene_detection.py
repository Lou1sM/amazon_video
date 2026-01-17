import os
import json
import cv2
import math
from time import time
from natsort import natsorted
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from dl_utils.misc import check_dir
from dl_utils.tensor_funcs import numpyify
import subprocess
import imageio_ffmpeg
import ffmpeg
import re
import pandas as pd
from dl_utils.misc import time_format
from os.path import join
from utils import segmentation_metrics, metric_names, bbc_mean_maxs, get_moviesumm_testnames, get_moviesumm_splitpoints
from simplified_segment import SceneSegmenter

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

class OldSceneSegmenter():
    def __init__(self, dset_name):
        self.dset_name = dset_name
        pass

    def get_ffmpeg_keyframe_times(self, vidname, recompute, avg_kf_every=5):
        starttime = time()
        self.framesdir = f'data/ffmpeg-keyframes/{vidname}'
        check_dir(self.framesdir)
        if (not os.path.isfile(join(self.framesdir, 'frametimes.npy'))) or recompute:
            for fn in os.listdir(self.framesdir):
                os.remove(join(self.framesdir, fn))
            print('extracting ffmpeg frames to', self.framesdir)
            vid_fpath = f"data/full-videos/{self.dset_name}/{vidname}.mp4"
            assert os.path.exists(vid_fpath), f'{vid_fpath} does not exist'
            duration = float(ffmpeg.probe(vid_fpath)["format"]["duration"])
            desired_nkfs = int(duration/avg_kf_every)
            x = subprocess.run([FFMPEG_PATH, "-i", vid_fpath, "-filter:v", "select=" "'gt(scene,0.05)'" ",showinfo" ",setpts=N/FR/TB", "-vsync", "0", f"{self.framesdir}/%05d.jpg"], capture_output=True)
            if 'Error submitting video frame to the encoder' in x.stderr.decode():
                x = subprocess.run([FFMPEG_PATH, "-i", vid_fpath, "-filter:v", "select=" "'gt(scene,0.1)'" ",showinfo", "-vsync", "0", f"{self.framesdir}/%05d.jpg"], capture_output=True)
            timepoint_lines = [z for z in x.stderr.decode().split('\n') if ' n:' in z]
            timepoints = np.array([float(re.search(r'(?<= pts_time:)[0-9\.]+(?= )',tl).group()) for tl in timepoint_lines])
            print('orig num kfs:', len(timepoint_lines), 'desired num kfs:', desired_nkfs)
            print('num feat files', len(os.listdir(f'data/ffmpeg-frame-features/{vidname}')))
            if len(timepoints)>desired_nkfs:
                keep_idxs = np.linspace(0,len(timepoints)-1, desired_nkfs).astype(int)
                for i in range(len(timepoints)):
                    if i not in keep_idxs:
                        os.remove(f'{self.framesdir}/{i:05d}.jpg')
                        if os.path.exists(maybe_feat_fp:=f'data/ffmpeg-frame-features/{vidname}/{i:05d}.npy'):
                            os.remove(maybe_feat_fp)
                            print('removing', f'data/ffmpeg-frame-features/{vidname}/{i:05d}.jpg','now n feat files is', len(os.listdir(f'data/ffmpeg-frame-features/{vidname}')))
                timepoints = timepoints[keep_idxs]
            np.save(join(self.framesdir, 'frametimes.npy'), timepoints)
            print(f'Keyframe extraction time: {time_format(time()-starttime)}')
        else:
            timepoints = np.load(join(self.framesdir, 'frametimes.npy'))
        if len(timepoints) != len(os.listdir(self.framesdir))-1:
            breakpoint()
        if len(timepoints) < len(os.listdir(f'data/ffmpeg-frame-features/{vidname}')):
            breakpoint()
        assert len(timepoints)+1 == len(os.listdir(self.framesdir)) # +1 for frametimes.npy
        return timepoints

    def cost_under_params(self, x, mu):
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        n, nz = x.shape
        mean_dists = x - mu
        #log_cov_det = np.log(sig).sum()
        mahala_dists = np.einsum('ij,ji->i', mean_dists/self.avg_sig, mean_dists.T)
        neg_log_probs = 0.5 * (nz*np.log(2*np.pi) + self.log_cov_det + mahala_dists)
        return neg_log_probs.mean(axis=0)*len(neg_log_probs) -neg_log_probs.max() + self.prec_cost*nz*(n-1)

    def cost_of_span(self, x, range_size):
        n, nz = x.shape
        sample_mu = x.mean(axis=0)
        #mean_cost = cost_under_params(sample_mu, arch_mu, avg_sig)
        precision_to_use = 2.4e-07 # smallest dist between points along any axis, hardcoding to save compute
        self.prec_cost = -np.log(precision_to_use)
        direct_cost = np.log(range_size) + self.prec_cost
        mean_cost = nz*direct_cost
        if n==1:
            #print(n, mean_cost)
            return mean_cost
        #sample_sig = x.var(axis=0) + 1e-7
        #var_cost = cost_under_params(sample_sig, arch_mu, avg_sig)
        data_cost = self.cost_under_params(x, sample_mu)
        cost = data_cost + mean_cost
        #print(n, data_cost, mean_cost, cost, cost/n)
        if np.isinf(cost):
            breakpoint()
        return cost

    def segment_from_feats_list(self, vidname, feats_list, recompute):
        N = len(feats_list)
        max_scene_size = 50
        splits_fp = check_dir('data/inferred-vid-splits')
        splits_fp = f'data/inferred-vid-splits/{vidname}-inferred-vid-splits.npy'
        if os.path.exists(splits_fp) and not recompute:
            print(f'loading splits from {splits_fp}')
            self.kf_scene_split_points = np.load(splits_fp)
        else:
            print('computing span costs')
            feat_vecs = np.stack(feats_list, axis=0)
            # avg_covariance in a length-20 chunk
            self.avg_sig = np.mean([feat_vecs[i*20:(i+1)*20].var(axis=0) for i in range(len(feat_vecs)//20)])
            self.log_cov_det = np.log(self.avg_sig).sum()
            range_size = feat_vecs.max() - feat_vecs.min()

            base_costs = [[np.inf if j<=i or j-i>=max_scene_size else self.cost_of_span(feat_vecs[i:j], range_size) for j in range(N+1)] for i in tqdm(range(N+1))]
            base_costs = np.array(base_costs)
            #base_costs = np.load('xmen-base-costs.npy')
            opt_costs = [base_costs[i,N-1] for i in range(N)]
            opt_splits = [[]]*N
            for start in reversed(range(N-1)):
                max_end = min(start+max_scene_size, N)
                opt_splits[start] = opt_splits[max_end-1]
                for possible_first_break in range(start+1, max_end):
                    maybe_new_cost = base_costs[start, possible_first_break] + opt_costs[possible_first_break]
                    if maybe_new_cost < opt_costs[start]:
                        opt_costs[start] = maybe_new_cost
                        opt_splits[start] = [possible_first_break] + opt_splits[possible_first_break]
                assert not np.isinf(opt_costs[start])

            self.kf_scene_split_points = opt_splits[0]
            np.save(splits_fp, self.kf_scene_split_points)
            print(f'found {len(self.kf_scene_split_points)+1} scenes')
        return self.kf_scene_split_points

    def scene_segment(self, vidname, recompute_keyframes, recompute_feats, recompute_best_split):
        timepoints = self.get_ffmpeg_keyframe_times(vidname, recompute=recompute_keyframes)
        fns = [x for x in os.listdir(self.framesdir) if x.endswith('.jpg')]
        assert len(timepoints) == len(fns)
        sorted_fns = natsorted(fns)

        check_dir(framefeatsdir:=f'data/ffmpeg-frame-features/{vidname}')
        if recompute_feats or any(not os.path.exists(f'{framefeatsdir}/{x.split(".")[0]}.npy') for x in fns):
            if not hasattr(self, 'model'):
                import open_clip
                self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
                self.model = self.model.cuda()

            self.model.eval()
        feat_paths = [f'{framefeatsdir}/{x.split(".")[0]}.npy' for x in sorted_fns]
        if recompute_feats or any(not os.path.exists(x) for x in feat_paths):
            print('extracting visual features')
            feats_list = []
            bs = 4
            batched = [(sorted_fns[i*bs:(i+1)*bs], feat_paths[i*bs:(i+1)*bs]) for i in range(int(math.ceil(len(sorted_fns)/bs)))]
            for i, (im_name_batch, featp_batch) in enumerate(tqdm(batched)):
                im_list = []
                for inb in im_name_batch:
                    image = Image.open(join(self.framesdir, inb))
                    breakpoint()
                    image = self.preprocess(image).unsqueeze(0).cuda()
                    im_list.append(image)
                image_batch = torch.cat(im_list)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    im_feats = self.model.encode_image(image_batch)
                    im_feats = numpyify(im_feats)
                batch_fps = feat_paths[i*bs:(i+1)*bs]
                for imf, ftp in zip(im_feats, batch_fps):
                    np.save(ftp, imf)
                    feats_list.append(imf)
        else:
            feats_list = [np.load(featp) for featp in feat_paths]

        self.segment_from_feats_list(vidname, feats_list, recompute=recompute_best_split)
        pt = np.array([timepoints[i] for i in self.kf_scene_split_points])
        #from dl_utils.misc import time_format
        #print('  '.join(time_format(x) for x in pt))
        return pt, timepoints

if __name__ == '__main__':

    import argparse
    from osvd_names import vidname2fps as osvd_vn2fps
    from get_bbc_times import annot_fns_list as bbc_annot_fns_list
    from osvd_names import get_scene_split_times as osvd_scene_split_times
    from get_bbc_times import get_scene_split_times as bbc_scene_split_times

    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-keyframes', action='store_true')
    parser.add_argument('--recompute-frame-features', action='store_true')
    parser.add_argument('--recompute-audio-features', action='store_true')
    parser.add_argument('--recompute-best-split', action='store_true')
    parser.add_argument('--recompute-all', action='store_true')
    parser.add_argument('--save-results', action='store_true')
    parser.add_argument('--use-audio', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--cut-first-n-secs', type=int, default=0)
    parser.add_argument('-d', '--dset', type=str, default='osvd')
    parser.add_argument('--kf-every', type=int, default=2)
    parser.add_argument('--subsample-frames', type=float, default=1)
    parser.add_argument('--feats-bs', type=int, default=4)
    parser.add_argument('--max-scene-len', type=int, default=600, help='maximum number of seconds that can be in a scene, lower is faster')
    parser.add_argument('--uniform-kfs', action='store_true')
    parser.add_argument('--model-name', type=str, choices=['clip', 'dinov2', 'vit', 'blip'], default='clip')
    parser.add_argument('--pow-incr', type=float, default=1.005, help='determines which points are skipped in search, higher is faster but less accurate')
    parser.add_argument('--use-avg-sig', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    ARGS = parser.parse_args()
    if ARGS.recompute_all:
        ARGS.recompute_keyframes = True
        ARGS.recompute_frame_features = True
        ARGS.recompute_best_split = True
        if not ARGS.no_save:
            ARGS.save_results = True

    device = 'cpu' if ARGS.cpu else 'cuda'
    def get_preds(pred_split_points, gt_n_scenes, avg_gt_scenes, ts):
        pred_point_labs = (np.expand_dims(ts,1)>pred_split_points).sum(axis=1)
        n_to_repeat = int(math.ceil(len(pred_point_labs)/avg_gt_scenes))
        n_to_repeat_oracle = int(math.ceil(len(pred_point_labs)/gt_n_scenes))
        unif_point_labs = np.repeat(np.arange(avg_gt_scenes), n_to_repeat)[:len(pred_point_labs)]
        unif_oracle_point_labs = np.repeat(np.arange(gt_n_scenes), n_to_repeat_oracle)[:len(pred_point_labs)]
        if len(unif_point_labs)!=len(pred_point_labs) or len(unif_oracle_point_labs)!=len(pred_point_labs):
            breakpoint()
        return pred_point_labs, unif_point_labs, unif_oracle_point_labs

    method_names = ['ours', 'uniform', 'uniform-oracle']
    x = 0
    max_seg_size = int(ARGS.max_scene_len/ARGS.kf_every)
    ss = SceneSegmenter(ARGS.dset, None, None, max_seg_size, ARGS.pow_incr, use_avg_sig=ARGS.use_avg_sig, kf_every=ARGS.kf_every, use_log_dist_cost=False, model_name=ARGS.model_name, use_audio=ARGS.use_audio, subsample_frames=ARGS.subsample_frames, device=device)
    if ARGS.dset=='osvd':
        avg_gt_scenes_dset = 22
        all_results = {m:{} for m in method_names}
        nframes = np.load('data/osvd-nframes.npy')
        for vn, fps in osvd_vn2fps.items():
            try:
                vid_fp = f'data/full-videos/{ARGS.dset}/{vn}.mp4'
                if isinstance(fps, str):
                    continue
                starttime = time()
                pred_split_points, all_timepoints = ss.scene_segment(vn, recompute_keyframes=ARGS.recompute_keyframes, recompute_frame_feats=ARGS.recompute_frame_features, recompute_audio_feats=ARGS.recompute_audio_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
                runtime = time()-starttime
                x += len(pred_split_points)
                ts = [t for t in all_timepoints if t > ARGS.cut_first_n_secs]
                k = int(len(ts)/(2*avg_gt_scenes_dset))
                ssts = osvd_scene_split_times(vn)
                gt = [x[1] for x in ssts[:-1]]
                pred_point_labs, unif_point_labs, unif_oracle_point_labs = get_preds(pred_split_points, len(ssts), avg_gt_scenes_dset, ts)
                gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
                for pred_name, preds in zip(method_names, [pred_point_labs, unif_point_labs, unif_oracle_point_labs]):
                    results = segmentation_metrics(gt_point_labs, preds, k=k)
                    results['runtime'] = runtime if pred_name=='ours' else 0
                    results['pred_n'] = len(pred_split_points) + 1
                    results['gt_n'] = len(gt) + 1
                    all_results[pred_name][vn] = results
            except torch.OutOfMemoryError:
                print('OOM for', vn)
        print(f'Mean predicted scenes: {x/len(all_results["ours"]):.3f}')
        combined = []
        for m in method_names:
            df = pd.DataFrame(all_results[m])
            df.loc['per-frame-runtime'] = df.loc['runtime']/nframes
            if m == 'ours':
                ours_df = df
            combined.append(df.mean(axis=1))
        results_df = pd.DataFrame(combined, index=method_names)
        results_df = results_df[metric_names]
        print(results_df)
        breakpoint()
        if ARGS.save_results:
            check_dir('segmentation-results/osvd')
        results_df.to_csv('segmentation-results/osvd/ours-unifs.csv')

    elif ARGS.dset=='bbc':
        avg_gt_scenes_dset = 48
        all_results_by_annot = [{m:{} for m in method_names} for _ in range(5)]
        nframes = np.load('data/bbc-nframes.npy')
        for vn in range(11):
            annotwise_ssts = bbc_scene_split_times(vn)
            starttime = time()
            pred_split_points, all_timepoints = ss.scene_segment(f'bbc_{vn+1:02}', recompute_keyframes=ARGS.recompute_keyframes, recompute_frame_feats=ARGS.recompute_frame_features, recompute_audio_feats=ARGS.recompute_audio_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
                #pred_split_points, all_timepoints = ss.scene_segment(vn, recompute_keyframes=ARGS.recompute_keyframes, recompute_frame_feats=ARGS.recompute_frame_features, recompute_audio_feats=ARGS.recompute_audio_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
            runtime = time()-starttime
            ts = [t for t in all_timepoints if t > ARGS.cut_first_n_secs]
            k = int(len(ts)/(2*avg_gt_scenes_dset))
            for annot_num, gt in enumerate(annotwise_ssts):
                pred_point_labs, unif_point_labs, unif_oracle_point_labs = get_preds(pred_split_points, len(gt)+1, avg_gt_scenes_dset, ts)
                gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
                x += len(gt)
                for i, (pred_name, preds) in enumerate(zip(method_names, [pred_point_labs, unif_point_labs, unif_oracle_point_labs])):
                    results = segmentation_metrics(gt_point_labs, preds, k=k)
                    results['runtime'] = runtime if pred_name=='ours' else 0
                    results['per-frame-runtime'] = runtime/nframes[i] if pred_name=='ours' else 0
                    results['pred_n'] = len(pred_split_points) + 1
                    results['gt_n'] = len(gt) + 1
                    #resu
                    #breakpoint()
                    all_results_by_annot[annot_num][pred_name][vn] = results
            #breakpoint()

        print(x//55, 'avg scenes')
        df=pd.json_normalize(all_results_by_annot)
        df.columns = pd.MultiIndex.from_tuples([tuple(col.split('.')) for col in df.columns])
        max_avgs, mean_avgs = bbc_mean_maxs(df)
        breakpoint()
        if ARGS.save_results:
            check_dir('segmentation-results/bbc')
            max_avgs.to_csv('segmentation-results/bbc-max/ours-unifs.csv')
            mean_avgs.to_csv('segmentation-results/bbc-mean/ours-unifs.csv')
        print('MAX')
        print(max_avgs)
        print('MEAN')
        print(mean_avgs)

    elif ARGS.dset=='moviesumm':
        ss = SceneSegmenter('moviesumm', None, None, max_seg_size, ARGS.pow_incr, use_avg_sig=False, kf_every=ARGS.kf_every, use_log_dist_cost=False, device=device)
        nframes = np.load('data/moviesumm-nframes.npy')
        test_vidnames, clean2cl = get_moviesumm_testnames()
        running_dict = {}
        avg_runtime = 0
        avg_pfr = 0
        all_results = {m:{} for m in method_names}
        avg_gt_scenes_dset = 53
        for i, vn in enumerate(pbar:=tqdm(test_vidnames)):
            try:
                starttime = time()
                pred_split_points, points_of_keyframes = ss.scene_segment(vn, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
                runtime = time()-starttime
                gt_split_points, betweens = get_moviesumm_splitpoints(vn)
                ts = np.load(f'data/ffmpeg-keyframes/moviesumm/{vn}/frametimes.npy')
                pred_point_labs, unif_point_labs, unif_oracle_point_labs = get_preds(pred_split_points, len(gt_split_points)+1, avg_gt_scenes_dset, ts)
                gt_point_labs = (np.expand_dims(ts,1)>gt_split_points).sum(axis=1)
                k = int(len(ts)/(2*avg_gt_scenes_dset))
                for pred_name, preds in zip(method_names, [pred_point_labs, unif_point_labs, unif_oracle_point_labs]):
                    results = segmentation_metrics(gt_point_labs, preds, k=k)
                    results['runtime'] = runtime if pred_name=='ours' else 0
                    all_results[pred_name][vn] = results
                per_frame_runtime = runtime/nframes[i]
                running_dict[vn] = {'runtime': runtime, 'per-frame-runtime': per_frame_runtime}
                avg_runtime = (runtime + (i*avg_runtime)) / (i+1)
                avg_pfr = (per_frame_runtime + (i*avg_pfr)) / (i+1)
                pbar.set_description(f'Avg runtime: {avg_runtime:.4f}, Per frame: {avg_pfr:.6f}')
                with open('running-dict.json', 'w') as f:
                    json.dump(running_dict, f)
            except torch.OutOfMemoryError:
                print('OOM for', vn)
        combined = []
        for m in method_names:
            df = pd.DataFrame(all_results[m])
            df.loc['per-frame-runtime'] = df.loc['runtime']/nframes[:df.shape[1]]
            combined.append(df)
        breakpoint()
        results_df = pd.DataFrame(combined, index=method_names)
        results_df = results_df[metric_names]
        print(results_df)
        if ARGS.save_results:
            check_dir('segmentation-results/osvd')
        results_df.to_csv('segmentation-results/osvd/ours-unifs.csv')
