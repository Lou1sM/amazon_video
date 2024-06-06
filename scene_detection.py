import os
from time import time
from natsort import natsorted
from tqdm import tqdm
import torch
from PIL import Image
import numpy as np
from dl_utils.misc import check_dir
from dl_utils.tensor_funcs import numpyify


#subprocess.run("ffmpeg -i SummScreen/videos/oltl-10-18-10.mp4        -filter:v "select='gt(scene,0.1)',showinfo"        -vsync 0 frames/%05d.jpg")

fns = [x for x in os.listdir('frames')]
sorted_fns = natsorted(fns)
#for i in range(len(sorted_fns)/5):
feats_list = []
check_dir('frame_features')
if any(not os.path.exists(f'frame_features/{x.split(".")[0]}.npy') for x in fns):
    print('some save paths don\'t exist, loading model')
    import open_clip
    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K')
    model = model.cuda()
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

for im_name in tqdm(sorted_fns):
    feats_fpath = f'frame_features/{im_name.split(".")[0]}.npy'
    if os.path.exists(feats_fpath):
        im_feats = np.load(feats_fpath)
    else:
        image = Image.open(f'frames/{im_name}')
        image = preprocess(image).unsqueeze(0).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            im_feats = model.encode_image(image)
            im_feats = numpyify(im_feats)
        np.save(feats_fpath, im_feats)
    feats_list.append(im_feats)

feat_vecs = np.concatenate(feats_list, axis=0)
#arch_mu = feat_vecs.mean(axis=0)
#arch_sig = feat_vecs.var(axis=0)+ 1e-7
arch_sig = np.mean([feat_vecs[i*20:(i+1)*20].var(axis=0) for i in range(len(feat_vecs)//20)])
log_cov_det = np.log(arch_sig).sum()
precision_to_use = 2.4e-07 # smallest dist between points along any axis, hardcoding to save compute
prec_cost = -np.log(precision_to_use)
range_size = feat_vecs.max() - feat_vecs.min()
direct_cost = np.log(range_size) + prec_cost

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
print('computing span costs')
#print(f'Time: {time_format(time()-start_time):.3f}')
base_costs = [[np.inf if j<=i or j-i>=max_scene_size else cost_of_span(feat_vecs[i:j])
                            for j in range(N)] for i in tqdm(range(N))]
base_costs = np.array(base_costs)
best_costs = np.empty([N,N])
best_splits = np.empty([N,N], dtype='object')
print('searching for optimal split')
for span_size in tqdm(range(N)):
    for start in range(N - span_size):
        stop = start+span_size#-1
        #if (start,stop) == (6,11):
            #breakpoint()
        no_split = base_costs[start,stop], []
        options = [no_split]
        #iter_to = min(stop, start+max_scene_size)
        iter_to = stop
        for k in range(start+1,iter_to):
            #if np.random.rand()>0.9:
                #assert (feat_vecs[start:stop] == np.concatenate([feat_vecs[start:k], feat_vecs[k:stop]], axis=0)).all()
                #assert cost_of_span(feat_vecs[start:k]) == base_costs[start,k]
                #assert cost_of_span(feat_vecs[k:stop]) == base_costs[k,stop]
            split_cost = best_costs[start,k] + best_costs[k,stop]
            split = best_splits[start,k] + [k] + best_splits[k,stop]
            options.append((split_cost,split))
        new_best_cost, new_best_splits = min(options, key=lambda x: x[0])
        best_costs[start,stop] = new_best_cost
        best_splits[start,stop] = new_best_splits
        #if len(new_best_splits)>2:
            #breakpoint()
print(best_splits[0,N-1])
print(best_costs[0,N-1])
breakpoint()

