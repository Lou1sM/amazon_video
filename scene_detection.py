import os
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
import re
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from dl_utils.label_funcs import accuracy as acc
from os.path import join
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--recompute-ffmpeg-scenes', action='store_true')
parser.add_argument('--recompute-frame-features', action='store_true')
parser.add_argument('--recompute-best-split', action='store_true')
parser.add_argument('--epname', type=str, default='oltl-10-18-10')
ARGS = parser.parse_args()



FFMEPG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
ARGS.epname = 'oltl-10-18-10'
check_dir(framesdir:=f'SummScreen/ffmpeg-scenes/{ARGS.epname}')
if (not os.path.isdir(framesdir)) or (not os.path.isfile(join(framesdir, 'frametimes.npy'))) or ARGS.recompute_ffmpeg_scenes:
    print('extracting ffmpeg frames to', framesdir)
    x = subprocess.run([FFMEPG_PATH, "-i", f"SummScreen/videos/{ARGS.epname}.mp4", "-filter:v", "select=" "'gt(scene,0.1)'" ",showinfo", "-vsync", "0", f"{framesdir}/%05d.jpg"], capture_output=True)
    timepoint_lines = [z for z in x.stderr.decode().split('\n') if ' n:' in z]
    timepoints = np.array([float(re.search(r'(?<= pts_time:)[0-9\.]+(?= )',tl).group()) for tl in timepoint_lines])
    np.save(join(framesdir, 'frametimes.npy'), timepoints)
else:
    timepoints = np.load(join(framesdir, 'frametimes.npy'))
fns = [x for x in os.listdir(framesdir) if x.endswith('.jpg')]
assert len(timepoints) == len(fns)
sorted_fns = natsorted(fns)
feats_list = []

check_dir(framefeatsdir:=f'SummScreen/ffmpeg-frame-features/{ARGS.epname}')
if ARGS.recompute_frame_features or any(not os.path.exists(f'{framefeatsdir}/{x.split(".")[0]}.npy') for x in fns):
    print('some save paths don\'t exist, loading model')
    import open_clip
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    model = model.cuda()
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

for im_name in tqdm(sorted_fns):
    feats_fpath = f'{framefeatsdir}/{im_name.split(".")[0]}.npy'
    if (not ARGS.recompute_frame_features) and os.path.exists(feats_fpath):
        im_feats = np.load(feats_fpath)
    else:
        image = Image.open(join(framesdir, im_name))
        image = preprocess(image).unsqueeze(0).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            im_feats = model.encode_image(image)
            im_feats = numpyify(im_feats)
        np.save(feats_fpath, im_feats)
    feats_list.append(im_feats)

def cost_under_params(x, mu):
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    assert x.ndim == 2
    n, nz = x.shape
    mean_dists = x - mu
    #log_cov_det = np.log(sig).sum()
    mahala_dists = np.einsum('ij,ji->i', mean_dists/arch_sig, mean_dists.T)
    neg_log_probs = 0.5 * (nz*np.log(2*np.pi) + log_cov_det + mahala_dists)
    return neg_log_probs.sum(axis=0) -neg_log_probs.max() + prec_cost*nz*(n-1)

def cost_of_span(x):
    n, nz = x.shape
    sample_mu = x.mean(axis=0)
    #mean_cost = cost_under_params(sample_mu, arch_mu, arch_sig)
    mean_cost = nz*direct_cost
    if n==1:
        #print(n, mean_cost)
        return mean_cost
    #sample_sig = x.var(axis=0) + 1e-7
    #var_cost = cost_under_params(sample_sig, arch_mu, arch_sig)
    data_cost = cost_under_params(x, sample_mu)
    cost = data_cost + mean_cost
    #print(n, data_cost, mean_cost, cost, cost/n)
    return cost

N = len(feats_list)
#N = 450
max_scene_size = 50
start_time = time()
splits_fp = check_dir('SummScreen/inferred-vid-splits')
splits_fp = f'SummScreen/inferred-vid-splits/{ARGS.epname}-inferred-vid-splits.npy'
if os.path.exists(splits_fp) and not ARGS.recompute_best_split:
    print(f'loading splits from {splits_fp}')
    ep_splits = np.load(splits_fp)
else:
    print('computing span costs')
    feat_vecs = np.concatenate(feats_list, axis=0)
    arch_sig = np.mean([feat_vecs[i*20:(i+1)*20].var(axis=0) for i in range(len(feat_vecs)//20)])
    log_cov_det = np.log(arch_sig).sum()
    precision_to_use = 2.4e-07 # smallest dist between points along any axis, hardcoding to save compute
    prec_cost = -np.log(precision_to_use)
    range_size = feat_vecs.max() - feat_vecs.min()
    direct_cost = np.log(range_size) + prec_cost

    base_costs = [[np.inf if j<=i or j-i>=max_scene_size else cost_of_span(feat_vecs[i:j])
                                for j in range(N)] for i in tqdm(range(N))]
    base_costs = np.array(base_costs)
    best_costs = np.empty([N,N])
    best_splits = np.empty([N,N], dtype='object')
    print('searching for optimal split')
    for span_size in tqdm(range(N)):
        for start in range(N - span_size):
            stop = start+span_size#-1
            no_split = base_costs[start,stop], []
            options = [no_split]
            iter_to = stop
            for k in range(start+1,iter_to):
                split_cost = best_costs[start,k] + best_costs[k,stop]
                split = best_splits[start,k] + [k] + best_splits[k,stop]
                options.append((split_cost,split))
            new_best_cost, new_best_splits = min(options, key=lambda x: x[0])
            best_costs[start,stop] = new_best_cost
            best_splits[start,stop] = new_best_splits
    ep_splits = np.array(best_splits[0,N-1])
    np.save(splits_fp, ep_splits)
pt = np.array([timepoints[i] for i in ep_splits])
gt_scene_times = pd.read_csv(f'SummScreen/video_scenes/{ARGS.epname}/startendtimes-from-transcript.csv')
gt = gt_scene_times['end'][:-1].to_numpy()
ts = np.arange(0,timepoints[-1],0.1)
gt_point_labs = (np.expand_dims(ts,1)>gt).sum(axis=1)
pred_point_labs = (np.expand_dims(ts,1)>pt).sum(axis=1)
print(ep_splits)
for mname ,mfunc in zip(['acc','nmi','ari'], [acc, nmi, ari]):
    score = mfunc(pred_point_labs, gt_point_labs)
    print(f'{mname}: {score:.4f}')
breakpoint()

