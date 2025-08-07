import os
from time import time
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
    def __init__(self, dset_name, show_name, season, max_seg_size, pow_incr, use_avg_sig, kf_every, use_log_dist_cost, device, model_name):
        self.dset_name = dset_name
        self.max_seg_size = max_seg_size
        self.pow_incr = pow_incr
        self.use_avg_sig = use_avg_sig
        self.kf_every = kf_every
        self.fps = 1/kf_every
        self.use_log_dist_cost = use_log_dist_cost
        self.device = device
        self.model_name = model_name
        self.feats_device = 'cuda'
        assert (show_name is None) == (season is None)
        if show_name is None:
            self.name_path = dset_name
        else:
            self.name_path = join(dset_name, show_name, f'season_{season}')
        self.vid_dir = join('data/full-videos', self.name_path)
        self.base_framesdir = join('data/ffmpeg-keyframes', self.name_path)
        self.base_featsdir =  join('data/ffmpeg-frame-features', self.name_path, self.model_name)
        self.cached_splits_dir = join('data/cached-splits', self.name_path, self.model_name)
        os.makedirs(self.base_framesdir, exist_ok=True)
        os.makedirs(self.base_featsdir, exist_ok=True)
        os.makedirs(self.cached_splits_dir, exist_ok=True)

    def get_ffmpeg_keyframe_times(self, vidname, recompute, uniform=False):
        framesdir = f'{self.base_framesdir}/{vidname}'
        os.makedirs(framesdir, exist_ok=True)
        vid_fpath = f"{self.vid_dir}/{vidname}.mp4"
        assert os.path.exists(vid_fpath), f'{vid_fpath} does not exist'
        duration = float(ffmpeg.probe(vid_fpath)["format"]["duration"])
        starttime = time()
        runtime_fp=f'ss-runtimes/frames/{vidname}'
        if os.path.isfile(frames_fp:=join(framesdir, 'frametimes.npy')) and os.path.isfile(runtime_fp) and (not recompute):
            timepoints = np.load(join(framesdir, 'frametimes.npy'))
        elif uniform:
            for fn in os.listdir(framesdir):
                os.remove(join(framesdir, fn))
            intdur = int(duration - 1/self.fps)
            end_timestamp = f"{intdur//3600:02}:{(intdur%3600)//60:02}:{intdur%60:02}"
            subprocess.run([FFMPEG_PATH, '-hide_banner', '-loglevel', 'error', '-ss', f'00:00:{int(1/self.fps):02}', '-to', end_timestamp, '-i', vid_fpath, '-vf', f'fps={self.fps}', f"{framesdir}/%05d.jpg"])
            nkfs = len(os.listdir(framesdir))
            if nkfs!= 1 + int((intdur - 1/self.fps -1) *self.fps):
                print(f'wrong nkfs: {nkfs}, when intdur={intdur}, fps={self.fps} so predicted nkfs is {1 + int((intdur - 1/self.fps) *self.fps)}')
            timepoints = np.linspace(1/self.fps, intdur, nkfs)
            np.save(join(framesdir, 'frametimes.npy'), timepoints)
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
            np.save(frames_fp, timepoints)
            runtime = time()-starttime
            os.makedirs('ss-runtimes/frames/', exist_ok=True)
            with open(runtime_fp, 'w') as f:
                f.write(str(runtime))
        if len(timepoints) != len(os.listdir(framesdir))-1:
            breakpoint()
        assert len(timepoints)+1 == len(os.listdir(framesdir)) # +1 for frametimes.npy
        return timepoints

    def load_model(self):
        if self.model_name == 'clip':
            import open_clip
            sys.stderr = io.StringIO()
            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
            sys.stderr = sys.__stderr__
            self.model = self.model.to(self.feats_device)
            self.model.eval()
            self.feat_fn = self.model.encode_image

        elif self.model_name == 'blip':
            from transformers import BlipProcessor, BlipForConditionalGeneration
            sys.stderr = io.StringIO()
            self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
            self.model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large').to(self.feats_device)
            sys.stderr = sys.__stderr__
            self.model.eval()
            self.feat_fn = lambda x: self.model.vision_model(pixel_values=x).last_hidden_state.mean(axis=1)
            self.preprocess = lambda x: torch.clamp(self.processor(x, return_tensors='pt').pixel_values.to(self.feats_device).squeeze(0), 0, 1)

        elif self.model_name == 'dinov2':
        # DINOv2 version
            import torchvision.transforms as transforms
            sys.stderr = io.StringIO()
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            sys.stderr = sys.__stderr__
            self.model = self.model.to(self.feats_device)
            self.model.eval()
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.feat_fn = self.model

        elif self.model_name == 'vit':
        # ViT with timm version
            import timm
            sys.stderr = io.StringIO()
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
            sys.stderr = sys.__stderr__
            self.model = self.model.to(self.feats_device)
            self.model.eval()
            data_config = timm.data.resolve_model_data_config(self.model)
            self.preprocess = timm.data.create_transform(**data_config, is_training=False)
            self.feat_fn = self.model
        else:
            sys.exit(f'unrecognised model name: {self.model_name}')

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

    def segment_by_means(self, feats, vidname):
        starttime = time()
        N = len(feats)
        max_seg_size = min(self.max_seg_size, N)
        x = np.unique(feats.flatten()).astype(float)
        self.precision_to_use = min(np.sort(x)[1:] - np.sort(x)[:-1])
        self.prec_cost = min(32, -np.log(self.precision_to_use))
        #max_segment_size = min(2000, N)
        feat_vecs = torch.tensor(feats, device=self.device, dtype=torch.float32)
        range_size = feat_vecs.max() - feat_vecs.min()

        direct_cost = torch.log(range_size) + self.prec_cost
        #batch_costs = torch.inf*torch.ones([N,N], device=self.device)
        batch_costs = torch.inf*torch.ones([N,N])
        batch_costs.diagonal()[:] = (feat_vecs.shape[1]*direct_cost).detach().cpu()
        bs = min(N, 20)
        if self.use_avg_sig:
            self.avg_sig = feat_vecs.unfold(0,bs,1).transpose(1,2).var(axis=1, keepdims=True).mean(axis=0)
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
                batch_costs.diagonal(offset=n+j-1)[:] = costs_to_set.detach().cpu()

        #base_costs = batch_costs.detach().cpu().numpy()
        base_costs = batch_costs.numpy()
        opt_splits, opt_cost = self.find_opt_cost(base_costs)
        if self.use_avg_sig:
            opt_cost += c_params_cost.item()

        os.makedirs('ss-runtimes/find-opt-cost/', exist_ok=True)
        with open(f'ss-runtimes/find-opt-cost/{vidname}', 'w') as f:
            f.write(str(time()-starttime))
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
            self.kf_scene_split_points = np.load(splits_fp).tolist()
        else:
            #print('computing span costs')
            with torch.no_grad():
                self.kf_scene_split_points = self.segment_by_means(feats_list, vidname)
            np.save(splits_fp, self.kf_scene_split_points)
        return self.kf_scene_split_points

    def scene_segment(self, vidname, recompute_keyframes, recompute_feats, recompute_best_split, bs, uniform_kfs):
        timepoints = self.get_ffmpeg_keyframe_times(vidname, recompute=recompute_keyframes, uniform=uniform_kfs)
        framesdir = f'{self.base_framesdir}/{vidname}'
        fns = [x for x in os.listdir(framesdir) if x.endswith('.jpg')]
        if not ( len(timepoints) == len(fns)):
            breakpoint()
        sorted_fns = natsorted(fns)

        os.makedirs(framefeatsdir:=os.path.join(self.base_featsdir, vidname), exist_ok=True)

        if recompute_feats or any(not os.path.exists(f'{framefeatsdir}/{x.split(".")[0]}.npy') for x in fns):
            if not hasattr(self, 'model'):
                self.load_model()

        feat_paths = [f'{framefeatsdir}/{x.split(".")[0]}.npy' for x in sorted_fns]
        starttime = time()
        runtime_fp=f'ss-runtimes/feats-extraction/{vidname}'
        if recompute_feats or any(not os.path.exists(x) for x in feat_paths) or (not os.path.exists(runtime_fp)):
            feats_list = []
            batched = [(sorted_fns[i*bs:(i+1)*bs], feat_paths[i*bs:(i+1)*bs]) for i in range(int(math.ceil(len(sorted_fns)/bs)))]
            for i, (im_name_batch, featp_batch) in enumerate(pbar:=tqdm(batched)):
                pbar.set_description(f'extracting frame features from {len(feat_paths)} frames')
                im_list = []
                for inb in im_name_batch:
                    image = Image.open(join(framesdir, inb))
                    image = self.preprocess(image).unsqueeze(0).to(self.feats_device)
                    im_list.append(image)
                image_batch = torch.cat(im_list)
                with torch.no_grad():
                    #im_feats = self.model.encode_image(image_batch)
                    im_feats = self.feat_fn(image_batch)
                    im_feats = im_feats.detach().cpu().numpy()
                batch_fps = feat_paths[i*bs:(i+1)*bs]
                for imf, ftp in zip(im_feats, batch_fps):
                    np.save(ftp, imf)
                    feats_list.append(imf)
        else:
            feats_list = [np.load(featp) for featp in feat_paths]

        try:
            feats = np.stack(feats_list, axis=0)
        except:
            breakpoint()
        os.makedirs('ss-runtimes/feats-extraction/', exist_ok=True)
        with open(runtime_fp, 'w') as f:
            f.write(str(time()-starttime))
        self.segment_from_feats_list(vidname, feats, recompute=recompute_best_split)
        pt = np.array([(timepoints[i]+timepoints[i+1])/2 for i in self.kf_scene_split_points])
        os.makedirs(kf_dir:=f'data/ffmpeg-keyframes-by-scene/{self.name_path}/{vidname}', exist_ok=True)
        next_scene_idx = 1
        np.save(join(kf_dir, 'scenesplit_timepoints.npy'), pt)
        np.save(join(kf_dir, 'scenesplit_idxs.npy'), self.kf_scene_split_points)
        for i, kf in enumerate(natsorted(os.listdir(framesdir))):
            if i in [0] + self.kf_scene_split_points:
                os.makedirs(cur_scene_dir:=f'{kf_dir}/scene{next_scene_idx}', exist_ok=True)
                for fn in os.listdir(cur_scene_dir): os.remove(join(cur_scene_dir, fn))
                next_scene_idx += 1
            if kf != 'frametimes.npy':
                os.symlink(os.path.abspath(f'{framesdir}/{kf}'), os.path.abspath(f'{cur_scene_dir}/{kf}'))
        print(len(pt))
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
    parser.add_argument('--show-name', type=str)
    parser.add_argument('--model-name', type=str, choices=['clip', 'dinov2', 'vit', 'blip'], default='clip')
    parser.add_argument('--season', type=str)
    parser.add_argument('--episode', type=str)
    parser.add_argument('--cpu', action='store_true')
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

    device = 'cpu' if ARGS.cpu else 'cuda'
    if ARGS.dset in ['osvd', 'bbc', 'moviesumm']:
        assert ARGS.show_name is None
        assert ARGS.season is None
        ARGS.episode = 'all'
    max_seg_size = int(ARGS.max_scene_len/ARGS.kf_every)
    print(max_seg_size)
    with torch.no_grad():
        if ARGS.episode == 'all':
            def segment_season(seas):
                print(f'splitting season {seas}')
                ss = SceneSegmenter(ARGS.dset, ARGS.show_name, seas, max_seg_size, ARGS.pow_incr, ARGS.use_avg_sig, ARGS.kf_every, use_log_dist_cost=ARGS.use_log_dist_cost, device=device, model_name=ARGS.model_name)
                for fname in natsorted(os.listdir(ss.vid_dir)):
                    vidname = fname.removesuffix('.mp4')
                    #if vidname in ['episode_5', 'episode_18']:
                    #if vidname != 'episode_7':
                        #continue
                    split_points, points_of_keyframes = ss.scene_segment(vidname, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
                    print(vidname, '  '.join(f'{int(sp//60)}m{sp%60:.1f}s' for sp in split_points))
            if ARGS.season == 'all':
                seasons = sorted([x.removeprefix('season_') for x in os.listdir(f'data/full-videos/tvqa/{ARGS.show_name}')])
                for s in seasons:
                    segment_season(s)
            else:
                segment_season(ARGS.season)
        else:
            assert ARGS.season != 'all'
            ss = SceneSegmenter(ARGS.dset, ARGS.show_name, ARGS.season, max_seg_size, ARGS.pow_incr, ARGS.use_avg_sig, ARGS.kf_every, use_log_dist_cost=ARGS.use_log_dist_cost, device=device, model_name=ARGS.model_name)
            split_points, points_of_keyframes = ss.scene_segment(f'episode_{ARGS.episode}', recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split, bs=ARGS.feats_bs, uniform_kfs=ARGS.uniform_kfs)
