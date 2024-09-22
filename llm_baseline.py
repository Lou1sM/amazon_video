import os
import json
from hierarchical_summarizer import load_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from utils import get_all_testnames
from dl_utils.misc import check_dir
from datasets import load_dataset


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-beams', type=int, default=3)
parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
parser.add_argument('--vidname', type=str, default='the-sixth-sense_1999')
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--with-script', action='store_true')
parser.add_argument('--with-whisper-transcript', action='store_true')
parser.add_argument('--mask-name', action='store_true')
parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
parser.add_argument('--expdir-prefix', type=str, default='experiments')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

assert not (ARGS.with_whisper_transcript and ARGS.with_script)
llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
            'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            }
model_name = llm_dict[ARGS.model]
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = load_peft_model(model_name, None, ARGS.prec)

if ARGS.mask_name:
    assert ARGS.with_script or ARGS.with_whisper_transcript

ds = load_dataset("rohitsaxena/MovieSum")
test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}
if ARGS.vidname != 'all':
    test_vidnames = [ARGS.vidname]

if ARGS.with_script:
    outdir = 'script-direct'
elif ARGS.with_whisper_transcript:
    outdir = 'with-whisper-direct'
else:
    outdir = 'vidname-only'

if ARGS.mask_name:
    outdir += '-masked-name'

if ARGS.model=='llama3-tiny':
    outdir += '-tiny'
elif ARGS.model=='llama3-8b':
    outdir += '-8b'

outdir = os.path.join(ARGS.expdir_prefix, outdir)

erroreds = []
for vn in tqdm(test_vidnames):
    if os.path.exists(maybe_summ_path:=f'{outdir}/{vn}-summary.txt') and not ARGS.recompute:
        print(f'Summ already at {maybe_summ_path}')
        continue

    if ARGS.with_script:
        gt_match_name = cl2clean[vn]
        gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
        gt_script = gt_match['script']
        summarize_prompt = f'Based on the following script:\n{gt_script}\nsummarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'
        #summarize_prompt = f'Summarize the movie {vn} based on the following script:\n{gt_script}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'

    elif ARGS.with_whisper_transcript:
        with open(f'data/transcripts/{vn}-no-names.json') as f:
           whisper_transcript = json.load(f)
        gt_match_name = cl2clean[vn]
        gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
        gt_script = gt_match['script']
        summarize_prompt = f'Based on the following transcript:\n{whisper_transcript}\nsummarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'

    else:
        summarize_prompt = f'Summarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'
    tok_ids = torch.tensor([tokenizer(summarize_prompt).input_ids])[:, -10000:].cuda()
    model.eval()
    n_beams = ARGS.n_beams
    min_len=600
    max_len=650
    with torch.no_grad():
        summ_tokens = None
        for n_tries in range(8):
            try:
                print('trying with', tok_ids.shape)
                summ_tokens = model.generate(tok_ids, min_new_tokens=min_len, max_new_tokens=max_len, num_beams=n_beams)
                break
            except torch.OutOfMemoryError:
                tok_ids = tok_ids[:,1000:]
                n_beams = max(1, n_beams-1)
                max_len -= 50
                min_len -= 50
                print(f'OOM, reducing min,max to {min_len}, {max_len}')
        if summ_tokens is None:
            erroreds.append(vn)
            break
        summ_tokens = summ_tokens[0,tok_ids.shape[1]:]
        summ = tokenizer.decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(summ)
        check_dir(outdir)
        print('writing to', maybe_summ_path)
        with open(maybe_summ_path, 'w') as f:
            f.write(summ)
