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
parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
            'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            }
model_name = llm_dict[ARGS.model]
#ep = episode_from_name(ARGS.vidname, False)
#transcript = '\n'.join(ep.scenes)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
if ARGS.vidname == 'all':
    #with open('moviesumm_testset_names.txt') as f:
    #    official_names = f.read().split('\n')
    #with open('clean-vid-names-to-command-line-names.json') as f:
    #    clean2cl = json.load(f)
    #assert all(x in official_names for x in clean2cl.keys())
    #test_vidnames = list(clean2cl.values())
    test_vidnames, clean2cl = get_all_testnames()
else:
    test_vidnames = [ARGS.vidname]
for vn in tqdm(test_vidnames):
    summarize_prompt = f'Summarize the movie {vn}. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Be sure to include information from all scenes, especially those at the end. Focus only on the plot events, no analysis or discussion of themes and characters.'
    tok_ids = torch.tensor([tokenizer(summarize_prompt).input_ids])
    model = load_peft_model(model_name, None, ARGS.prec)
    model.eval()
    with torch.no_grad():
        attention_mask = torch.ones_like(tok_ids)
        summ_tokens = model.generate(tok_ids, min_new_tokens=600, max_new_tokens=650, num_beams=ARGS.n_beams)
        #summ_tokens = model.generate(tok_ids, min_new_tokens=6, max_new_tokens=6, num_beams=ARGS.n_beams)
        summ_tokens = summ_tokens[0,tok_ids.shape[1]:]
        summ = tokenizer.decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(summ)
        check_dir('baseline-summs')
        with open('baseline-summs/{vn}.txt', 'w') as f:
            f.write(summ)
