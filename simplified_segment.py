import os
import io
import sys
import math
from natsort import natsorted
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
import subprocess
import imageio_ffmpeg
import ffmpeg
import re
from os.path import join

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
LOG2PI = 1.837877

class SceneSegmenter():
    def __init__(self, dset_name, show_name, season, max_seg_size, pow_incr, use_avg_sig, kf_every, use_log_dist_cost):
        self.dset_name = dset_name
        self.max_seg_size = max_seg_size
        self.pow_incr = pow_incr
        self.use_avg_sig = use_avg_sig
        self.kf_every = kf_every
        self.fps = 1/kf_every
        self.use_log_dist_cost = use_log_dist_cost
        assert (show_name is None) == (season is None)
        if show_name is None:
            self.name_path = dset_name
        else:
            self.name_path = join(dset_name, show_name, f'season_{season}')
        self.vid_dir = join('data/full-videos', self.name_path)
        self.base_framesdir = join('data/ffmpeg-keyframes', self.name_path)
        self.base_featsdir =  join('data/ffmpeg-frame-features', self.name_path)
        self.cached_splits_dir = join('data/cached-splits', self.name_path)
        os.makedirs(self.base_framesdir, exist_ok=True)
        os.makedirs(self.base_featsdir, exist_ok=True)
        os.makedirs(self.cached_splits_dir, exist_ok=True)

    def get_ffmpeg_keyframe_times(self, vidname, recompute, uniform=False):
        framesdir = f'{self.base_framesdir}/{vidname}'
        os.makedirs(framesdir, exist_ok=True)
        vid_fpath = f"{self.vid_dir}/{vidname}.mp4"
        assert os.path.exists(vid_fpath), f'{vid_fpath} does not exist'
        duration = float(ffmpeg.probe(vid_fpath)["format"]["duration"])
        if uniform:
            for fn in os.listdir(framesdir):
                os.remove(join(framesdir, fn))
            intdur = int(duration - 1/self.fps)
            end_timestamp = f"{intdur//3600:02}:{(intdur%3600)//60:02}:{intdur%60:02}"
            subprocess.run([FFMPEG_PATH, '-hide_banner', '-loglevel', 'error', '-ss', f'00:00:{int(1/self.fps):02}', '-to', end_timestamp, '-i', vid_fpath, '-vf', f'fps={self.fps}', f"{framesdir}/%05d.jpg"])
            #subprocess.run([FFMPEG_PATH, '-ss', f'00:00:{int(1/self.fps):02}', '-to', end_timestamp, '-i', vid_fpath, '-vf', f'fps={self.fps}', f"{framesdir}/%05d.jpg"])
            nkfs = len(os.listdir(framesdir))
            if nkfs!= 1 + int((intdur - 1/self.fps -1) *self.fps):
                print(f'wrong nkfs: {nkfs}, when intdur={intdur}, fps={self.fps} so predicted nkfs is {1 + int((intdur - 1/self.fps) *self.fps)}')
            #timepoints = np.arange(1/self.fps, intdur, 1/self.fps)
            timepoints = np.linspace(1/self.fps, intdur, nkfs)
            np.save(join(framesdir, 'frametimes.npy'), timepoints)
        elif os.path.isfile(join(framesdir, 'frametimes.npy')) and (not recompute):
            timepoints = np.load(join(framesdir, 'frametimes.npy'))
        else:
            desired_nkfs = int(duration * self.fps)
            for fn in os.listdir(framesdir):
                os.remove(join(framesdir, fn))
            print('extracting ffmpeg frames to', framesdir)
            x = subprocess.run([FFMPEG_PATH, "-i", vid_fpath, "-filter:v", "select=" "'gt(scene,0.05)'" ",showinfo" ",setpts=N/FR/TB", "-vsync", "0", f"{framesdir}/%05d.jpg"], capture_output=True)
            if 'Error submitting video frame to the encoder' in x.stderr.decode():
                x = subprocess.run([FFMPEG_PATH, "-i", vid_fpath, "-filter:v", "select=" "'gt(scene,0.1)'" ",showinfo", "-vsync", "0", f"{framesdir}/%05d.jpg"], capture_output=True)
            timepoint_lines = [z for z in x.stderr.decode().split('\n') if ' n:' in z]
            timepoints = np.array([float(re.search(r'(?<= pts_time:)[0-9\.]+(?= )',tl).group()) for tl in timepoint_lines])
            print('orig num kfs:', len(timepoint_lines), 'desired num kfs:', desired_nkfs)
            if len(timepoints)>desired_nkfs:
                keep_idxs = np.linspace(0,len(timepoints)-1, desired_nkfs).astype(int)
                for i in range(len(timepoints)):
                    if i not in keep_idxs:
                        os.remove(f'{framesdir}/{i:05d}.jpg')
                        if os.path.exists(maybe_feat_fp:=f'{self.base_framesdir}/{vidname}/{i:05d}.npy'):
                            os.remove(maybe_feat_fp)
                            print('removing', f'{self.base_framesdir}/{vidname}/{i:05d}.jpg','now n feat files is', len(os.listdir(f'{self.base_framesdir}/{vidname}')))
                timepoints = timepoints[keep_idxs]
            np.save(join(framesdir, 'frametimes.npy'), timepoints)
        if len(timepoints) != len(os.listdir(framesdir))-1:
            breakpoint()
        assert len(timepoints)+1 == len(os.listdir(framesdir)) # +1 for frametimes.npy
        return timepoints

    def batched_cost_under_params(self, x, mu, sigs):
        N, n, nz = x.shape
        mean_dists = x - mu
        if sigs is None:
            sigs = self.avg_sig
            log_cov_det = self.log_cov_det
        else:
            log_cov_det = sigs.log().sum(axis=2)
        if self.use_log_dist_cost:
            costs = (abs(mean_dists).log() + 1)
            costs[mean_dists==0] = 0
            prec_cost = max(-costs.min(), self.prec_cost)
            costs += prec_cost
            if not ( costs.min() >= 0):
               breakpoint()
            costs = costs.sum(axis=2).sum(axis=1)
        else:
            mahala_dists = ((mean_dists/sigs) * mean_dists).sum(axis=2)
            neg_log_probs = 0.5 * (nz*LOG2PI + log_cov_det + mahala_dists)
            costs = neg_log_probs.sum(axis=1) - neg_log_probs.max(axis=1).values  + self.prec_cost*nz*(n-1)
        if torch.isinf(costs).any() or torch.isnan(costs).any():
            breakpoint()
        return costs

    def cost_of_span(self, x, range_size):
        n, nz = x.shape
        mu = x.mean(axis=0)
        sig = x.var(axis=0)
        direct_cost = np.log(range_size) + self.prec_cost
        mean_cost = nz*direct_cost
        if n==1:
            return mean_cost
        data_cost = self.cost_under_params(x, mu, sig, np.log(sig).sum())
        cost = data_cost + mean_cost
        if np.isinf(cost):
            breakpoint()
        return cost

    def cost_under_params(self, x, mu, sig, log_cov_det):
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2
        n, nz = x.shape
        mean_dists = x - mu
        mahala_dists = np.einsum('ij,ji->i', mean_dists/sig, mean_dists.T)
        neg_log_probs = 0.5 * (nz*LOG2PI + log_cov_det + mahala_dists)
        cost = neg_log_probs.mean(axis=0)*len(neg_log_probs) -neg_log_probs.max() + self.prec_cost*nz*(n-1) # mean not sum helps overflow
        #assert (allclose(cost, neg_log_probs.sum(axis=0) - neg_log_probs.max() + self.prec_cost*nz*(n-1)))
        return cost

    def segment_by_means(self, feats):
        N = len(feats)
        max_seg_size = min(self.max_seg_size, N)
        x = np.unique(feats.flatten()).astype(float)
        self.precision_to_use = min(np.sort(x)[1:] - np.sort(x)[:-1])
        self.prec_cost = min(32, -np.log(self.precision_to_use))
        #max_segment_size = min(2000, N)
        feat_vecs = torch.tensor(feats, device='cuda', dtype=torch.float32)
        range_size = feat_vecs.max() - feat_vecs.min()

        direct_cost = torch.log(range_size) + self.prec_cost
        batch_costs = torch.inf*torch.ones([N,N], device='cuda')
        batch_costs.diagonal()[:] = feat_vecs.shape[1]*direct_cost
        if self.use_avg_sig:
            self.avg_sig = feat_vecs.unfold(0,20,1).transpose(1,2).var(axis=1, keepdims=True).mean(axis=0)
            self.log_cov_det = self.avg_sig.log().sum()
        if self.pow_incr==1:
            sizes_to_compute = np.arange(2,max_seg_size)
        else:
            sizes_to_compute = sorted(set([int(self.pow_incr**i) for i in range(int(math.ceil(np.log(2)/np.log(self.pow_incr))), int(np.log(max_seg_size)/np.log(self.pow_incr)))])) + [max_seg_size+1]
        to_enum = enumerate(sizes_to_compute[:-1])
        if N > 1000:
            to_enum = tqdm(list(to_enum))
        for i,n in to_enum:
            c_params_cost = feat_vecs.shape[1]*direct_cost
            stacked = feat_vecs.unfold(0,n,1).transpose(1,2)
            means = stacked.mean(axis=1, keepdims=True)
            if self.use_avg_sig:
                sigs = None
            else:
                sigs = stacked.var(axis=1, keepdims=True)
                sigs[sigs==0] += 1e-7
            residual_cost = self.batched_cost_under_params(stacked, means, sigs)
            costs = residual_cost + c_params_cost if self.use_avg_sig else residual_cost + 2*c_params_cost
            assert len(costs) == N-n+1
            if not ( len(costs) == N-n+1):
                breakpoint()
            next_n = sizes_to_compute[i+1]
            for j in range(next_n-n):
                costs_to_set = torch.stack([costs[k:len(costs)-j+k] for k in range(j+1)]).mean(axis=0)
                if not ( len(costs_to_set)==len(costs)-j):
                    breakpoint()
                batch_costs.diagonal(offset=n+j-1)[:] = costs_to_set

        base_costs = batch_costs.detach().cpu().numpy()
        opt_splits, opt_cost = self.find_opt_cost(base_costs)
        if self.use_avg_sig:
            opt_cost += c_params_cost.item()
        return opt_splits

    def find_opt_cost(self, base_costs):
        N = base_costs.shape[0]
        opt_costs = [base_costs[i,N-1] for i in range(N)]
        opt_splits = [[]]*N
        for start in reversed(range(N-1)):
            max_end = min(start+self.max_seg_size, N)
            opt_splits[start] = opt_splits[max_end-1]
            for possible_first_break in range(start, max_end-1):
                maybe_new_cost = base_costs[start, possible_first_break] + opt_costs[possible_first_break+1]# + np.log2(N)
                if maybe_new_cost < opt_costs[start]:
                    opt_costs[start] = maybe_new_cost
                    opt_splits[start] = [possible_first_break] + opt_splits[possible_first_break+1]
            if not ( not np.isinf(opt_costs[start])):
                breakpoint()

        return opt_splits[0], opt_costs[0]

    def segment_from_feats_list(self, vidname, feats_list, recompute):
        splits_fp = f'{self.cached_splits_dir}/{vidname}-inferred-vid-splits.npy'
        if os.path.exists(splits_fp) and not recompute:
            #print(f'loading splits from {splits_fp}')
            self.kf_scene_split_points = np.load(splits_fp)
        else:
            #print('computing span costs')
            self.kf_scene_split_points = self.segment_by_means(feats_list)
        return self.kf_scene_split_points

    def scene_segment(self, vidname, recompute_keyframes, recompute_feats, recompute_best_split, bs, uniform_kfs):
        timepoints = self.get_ffmpeg_keyframe_times(vidname, recompute=recompute_keyframes, uniform=uniform_kfs)
        framesdir = f'{self.base_framesdir}/{vidname}'
        fns = [x for x in os.listdir(framesdir) if x.endswith('.jpg')]
        if not ( len(timepoints) == len(fns)):
            breakpoint()
        sorted_fns = natsorted(fns)

        os.makedirs(framefeatsdir:=f'{self.base_featsdir}/{vidname}', exist_ok=True)

        if recompute_feats or any(not os.path.exists(f'{framefeatsdir}/{x.split(".")[0]}.npy') for x in fns):
            if not hasattr(self, 'model'):
                import open_clip
                sys.stderr = io.StringIO()
                self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
                sys.stderr = sys.__stderr__
                self.model = self.model.cuda()
                self.model.eval()

        feat_paths = [f'{framefeatsdir}/{x.split(".")[0]}.npy' for x in sorted_fns]
        if recompute_feats or any(not os.path.exists(x) for x in feat_paths):
            feats_list = []
            batched = [(sorted_fns[i*bs:(i+1)*bs], feat_paths[i*bs:(i+1)*bs]) for i in range(int(math.ceil(len(sorted_fns)/bs)))]
            for i, (im_name_batch, featp_batch) in enumerate(pbar:=tqdm(batched)):
                pbar.set_description(f'extracting frame features from {len(feat_paths)} frames')
                im_list = []
                for inb in im_name_batch:
                    image = Image.open(join(framesdir, inb))
                    image = self.preprocess(image).unsqueeze(0).cuda()
                    im_list.append(image)
                image_batch = torch.cat(im_list)
                with torch.no_grad():
                    im_feats = self.model.encode_image(image_batch)
                    im_feats = im_feats.detach().cpu().numpy()
                batch_fps = feat_paths[i*bs:(i+1)*bs]
                for imf, ftp in zip(im_feats, batch_fps):
                    np.save(ftp, imf)
                    feats_list.append(imf)
        else:
            feats_list = [np.load(featp) for featp in feat_paths]

        feats = np.stack(feats_list, axis=0)
        self.segment_from_feats_list(vidname, feats, recompute=recompute_best_split)
        pt = np.array([timepoints[i] for i in self.kf_scene_split_points])
        kf_dir = f'data/ffmpeg-keyframes-by-scene/{self.name_path}'
        #os.makedirs(cur_scene_dir:=f'{kf_dir}/scene0', exist_ok=True)
        next_scene_idx = 1
        for i, kf in enumerate(natsorted(os.listdir(framesdir))):
            if i in [0] + self.kf_scene_split_points:
                os.makedirs(cur_scene_dir:=f'{kf_dir}/scene{next_scene_idx}', exist_ok=True)
                for fn in os.listdir(cur_scene_dir): os.remove(join(cur_scene_dir, fn))
                next_scene_idx += 1
            if kf != 'frametimes.npy':
                os.symlink(os.path.abspath(f'{framesdir}/{kf}'), os.path.abspath(f'{cur_scene_dir}/{kf}'))
        return pt, timepoints

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-keyframes', action='store_true')
    parser.add_argument('--recompute-frame-features', action='store_true')
    parser.add_argument('--recompute-best-split', action='store_true')
    parser.add_argument('--recompute-all', action='store_true')
    #parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--dset', '-d', type=str, required=True)
    parser.add_argument('--show-name', type=str, required=True)
    parser.add_argument('--season', type=str, required=True)
    parser.add_argument('--feats-bs', type=int, default=4, help='batch size for extracting features')
    parser.add_argument('--max-scene-len', type=int, default=240, help='maximum number of seconds that can be in a scene, lower is faster')
    parser.add_argument('--pow-incr', type=float, default=1.005, help='determines which points are skipped in search, higher is faster but less accurate')
    parser.add_argument('--use-avg-sig', action='store_true')
    parser.add_argument('--uniform-kfs', action='store_true')
    parser.add_argument('--use-log-dist-cost', action='store_true')
    parser.add_argument('--kf-every', type=int, default=2)
    ARGS = parser.parse_args()
    if ARGS.recompute_all:
        ARGS.recompute_keyframes = True
        ARGS.recompute_frame_features = True
        ARGS.recompute_best_split = True

    if ARGS.dset in ['osvd', 'bbc', 'moviesumm']:
        assert ARGS.show_name is None
        assert ARGS.season_name is None
    max_seg_size = int(ARGS.max_scene_len/ARGS.kf_every)
    print(max_seg_size)
    ss = SceneSegmenter(ARGS.dset, ARGS.show_name, ARGS.season, max_seg_size, ARGS.pow_incr, ARGS.use_avg_sig, ARGS.kf_every, use_log_dist_cost=ARGS.use_log_dist_cost)
    for fname in os.listdir(ss.vid_dir):
        vidname = fname.removesuffix('.mp4')
        #if vidname in ['episode_5', 'episode_18']:
        if vidname != 'episode_7':
            continue
        split_points, points_of_keyframes = ss.scene_segment(vidname, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
        print(vidname, '  '.join(f'{int(sp//60)}m{sp%60:.1f}s' for sp in split_points))
