import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from utils import get_all_testnames
from datasets import load_dataset
from os.path import join
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (get_model_name_from_path,
                           tokenizer_image_token,
                           KeywordsStoppingCriteria)
from llava.constants import (IMAGE_TOKEN_INDEX,
                           DEFAULT_IMAGE_TOKEN,
                           DEFAULT_IM_START_TOKEN,
                           DEFAULT_IM_END_TOKEN)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-beams', type=int, default=3)
parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
parser.add_argument('--min-len', type=int, default=600)
parser.add_argument('--max-len', type=int, default=650)
parser.add_argument('--vidname', type=str, default='the-sixth-sense_1999')
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--with-script', action='store_true')
parser.add_argument('--with-whisper-transcript', action='store_true')
parser.add_argument('--with-caps', action='store_true')
parser.add_argument('--mask-name', action='store_true')
parser.add_argument('--no-model', action='store_true')
parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
parser.add_argument('--expdir-prefix', type=str, default='experiments')
parser.add_argument('--kf-dir-prefix', type=str, default='experiments')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

assert not (ARGS.with_whisper_transcript and ARGS.with_script)
overwrite_config = {"mm_spatial_pool_mode": 'average', "mm_spatial_pool_stride": 4, "mm_newline_position": 'no_token'}
if ARGS.no_model:
    tokenizer, model = None, None
else:
    tokenizer, model, processor, _ = load_pretrained_model(
        'lmms-lab/LLaVA-Video-7B-Qwen2',
        None,
        'LLaVA-Video-7B-Qwen2',
        load_8bit=False, overwrite_config=overwrite_config, attn_implementation='sdpa')

if ARGS.device=='cpu':
    model.to('cpu')
if ARGS.mask_name:
    assert ARGS.with_script or ARGS.with_whisper_transcript

ds = load_dataset("rohitsaxena/MovieSum")
test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}
if ARGS.vidname != 'all':
    test_vidnames = [ARGS.vidname]

if ARGS.with_script:
    outdir = 'script-direct'
else:
    outdir = 'vidname-only'

if ARGS.mask_name:
    outdir += '-masked-name'

outdir = os.path.join(ARGS.expdir_prefix, outdir)

erroreds = []
dset_name = 'moviesumm'
for vn in tqdm(test_vidnames):
    vid_subpath = join(dset_name, vn)
    if os.path.exists(maybe_summ_path:=f'{outdir}/{vn}-summary.txt') and not ARGS.recompute:
        print(f'Summ already at {maybe_summ_path}')
        continue

    if ARGS.with_script:
        gt_match_name = cl2clean[vn]
        gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
        gt_script = gt_match['script']
        summarize_prompt = f'Based on the following script:\n{gt_script}\nsummarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'

        from torchvision.transforms import ToTensor
        transform = ToTensor()
        image_dir = join(ARGS.kf_dir_prefix, 'ffmpeg-keyframes', vid_subpath)
        image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]
        max_frames_num = 4  # Reduced for memory efficiency
        pick_every = len(image_paths) // max_frames_num
        image_paths = image_paths[::pick_every][:max_frames_num]
        assert len(image_paths) == max_frames_num

        images = [Image.open(fp) for fp in image_paths]
        images = [transform(np.array(x)).half() for x in images]
        images = [x[:, :384, :384] for x in images]
        images = [x.to(model.device) for x in images]
    else:
        summarize_prompt = f'Summarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'
        images = [torch.zeros(3, 384, 384).half().to(model.device)]
    input_ids = tokenizer_image_token(summarize_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(ARGS.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids,images=[x[:, :384, :384] for x in images] ,modalities=["video"],do_sample=False,temperature=0,max_new_tokens=1,)
    tok_ids = torch.tensor([tokenizer(summarize_prompt).input_ids])[:, -10000:].to(model.device)
    model.eval()
    n_beams = ARGS.n_beams
    #ARGS.min_len=600
    #ARGS.max_len=650
    with torch.no_grad():
        summ_tokens = None
        for n_tries in range(8):
            try:
                print('trying with', tok_ids.shape)
                summ_tokens = model.generate(tok_ids, min_new_tokens=ARGS.min_len, max_new_tokens=ARGS.max_len, num_beams=n_beams)
                break
            except torch.OutOfMemoryError:
                tok_ids = tok_ids[:,1000:]
                n_beams = max(1, n_beams-1)
                ARGS.max_len -= 50
                ARGS.min_len -= 50
                print(f'OOM, reducing min,max to {ARGS.min_len}, {ARGS.max_len}')
        if summ_tokens is None:
            erroreds.append(vn)
            break
        summ_tokens = summ_tokens[0,tok_ids.shape[1]:]
        summ = tokenizer.decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(summ)
        os.makedirs(outdir, exist_ok=True)
        print('writing to', maybe_summ_path)
        with open(maybe_summ_path, 'w') as f:
            f.write(summ)

print(erroreds)
with open('errored.txt', 'w') as f:
    f.write('\n'.join(erroreds))
