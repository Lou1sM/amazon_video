import os
from summarize_dialogue import load_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from utils import get_all_testnames
from dl_utils.misc import check_dir


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n-beams', type=int, default=3)
parser.add_argument('--prec', type=int, default=32, choices=[32,8,4,2])
parser.add_argument('--vidname', type=str, default='the-sixth-sense_1999')
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
            'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            }
model_name = llm_dict[ARGS.model]
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if ARGS.vidname == 'all':
    test_vidnames, clean2cl = get_all_testnames()
else:
    test_vidnames = [ARGS.vidname]
for vn in tqdm(test_vidnames):
    if os.path.exists(maybe_summ_path:=f'baseline-summs/{vn}.txt') and not ARGS.recompute:
        print(f'Summ already at {maybe_summ_path}')
    summarize_prompt = f'Summarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Focus only on the plot events, no analysis or discussion of themes and characters.'
    tok_ids = torch.tensor([tokenizer(summarize_prompt).input_ids])
    model = load_peft_model(model_name, None, ARGS.prec)
    model.eval()
    n_beams = ARGS.n_beams
    min_len=600
    max_len=650
    with torch.no_grad():
        attention_mask = torch.ones_like(tok_ids)
        for i in range(8):
            try:
                summ_tokens = model.generate(tok_ids, min_new_tokens=min_len, max_new_tokens=max_len, num_beams=n_beams)
                break
            except torch.OutOfMemoryError:
                summarize_prompt = summarize_prompt[:-30]
                n_beams = max(1, n_beams-1)
                max_len -= 50
                min_len -= 50
                print(f'OOM, reducing min,max to {min_len}, {max_len}')
        summ_tokens = summ_tokens[0,tok_ids.shape[1]:]
        summ = tokenizer.decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(summ)
        check_dir('baseline-summs')
        with open(maybe_summ_path, 'w') as f:
            f.write(summ)
