import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dl_utils.misc import set_experiment_dir
from datasets import load_dataset, load_from_disk
import argparse
import torch
import os
from os.path import join
from summarize_dialogue import SoapSummer
import sys
from utils import get_fn


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions'], default='nocaptions')
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--reorder', action='store_true')
parser.add_argument('--randorder', action='store_true')
parser.add_argument('--resumm_scenes',action='store_true')
parser.add_argument('--eval_every',type=int,default=1)
parser.add_argument('--eval_first',action='store_true')
parser.add_argument('--expname',type=str)
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn')
parser.add_argument('--n_dpoints',type=int,default=-1)
parser.add_argument('--bs',type=int,default=1)
parser.add_argument('--dbs',type=int,default=8)
parser.add_argument('--n_epochs',type=int,default=2)
parser.add_argument('--n_iter',type=int,default=-1)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--reload_from',type=str,default='none')
parser.add_argument('--retokenize',action='store_true')
parser.add_argument('--save_every',action='store_true')
parser.add_argument('--dont_save_new_scenes',action='store_true')
parser.add_argument('-bt','--is_test_big_bart',action='store_true')
parser.add_argument('-t','--is_test',action='store_true')
ARGS = parser.parse_args()


assert not (ARGS.reorder and ARGS.randorder)
ARGS.is_test = ARGS.is_test or ARGS.is_test_big_bart
ARGS.retokenize = ARGS.retokenize or ARGS.resumm_scenes

if ARGS.expname is None and not ARGS.is_test:
    sys.exit('set a different expname')
elif ARGS.is_test:
    ARGS.expname='tmp'

print(ARGS.expname)


expname = set_experiment_dir(ARGS.expname, ARGS.overwrite, name_of_trials='tmp')

device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

model_name = 'lucadiliello/bart-small' if ARGS.is_test and not ARGS.is_test_big_bart else ARGS.model_name

print(f'using model {model_name}')

if ARGS.reload_from=='none':
    chkpt_path = model_name
else:
    chkpt_path = f'./experiments/{ARGS.expname}/checkpoints/epoch{ARGS.reload_from}'

print(f'loading from {chkpt_path}')
model = AutoModelForSeq2SeqLM.from_pretrained(chkpt_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(chkpt_path)
scene_order = 'optimal' if ARGS.reorder else 'rand' if ARGS.randorder else 'identity'
ss = SoapSummer(model=model,
                bs=ARGS.bs,
                dbs=ARGS.dbs,
                tokenizer=tokenizer,
                caps=ARGS.caps,
                reorder=ARGS.reorder,
                randorder=ARGS.randorder,
                expname=expname,
                resumm_scenes=ARGS.resumm_scenes,
                do_save_new_scenes=not ARGS.dont_save_new_scenes,
                is_test=ARGS.is_test)

fn = get_fn(ARGS.caps, ARGS.reorder, ARGS.randorder, ARGS.is_test, ARGS.n_dpoints)
if os.path.exists(f'SummScreen/cached_tokenized/{fn}_train_cache') and os.path.exists('SummScreen/cached_tokenized/{fn}_test_cache') and not ARGS.retokenize:
    print('tokenized datasets have been cached, loading')
    tokenized_trainset = load_from_disk('cached_trainset')
    tokenized_testset = load_from_disk('cached_testset')
else:
    path_to_json_trainset = f'SummScreen/json_datasets/train_{fn}_dset.json'
    print(f'json trainset path is {path_to_json_trainset}')
    path_to_json_testset = path_to_json_trainset.replace('train','test')
    # sharding isn't supported atm so everything ends up in 'test'
    if not os.path.exists(path_to_json_trainset) or not os.path.exists(path_to_json_testset) or ARGS.retokenize:
        print('building new dataset')
        ss.build_dset(ARGS.caps, ARGS.n_dpoints)
        assert os.path.exists(path_to_json_trainset)
        assert os.path.exists(path_to_json_testset)

    trainset = load_dataset('json', data_files=path_to_json_trainset,split='train')
    testset = load_dataset('json', data_files=path_to_json_testset,split='train')

    def train_preprocess_function(dpoint):
        inputs = [doc for doc in dpoint['scene_summs']]
        model_inputs = tokenizer(inputs)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=dpoint['summ'])

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    def test_preprocess_function(dpoint):
        model_inputs = tokenizer(dpoint.pop('scene_summs'))
        for k,v in dpoint.items():
            model_inputs[k] = v
        return model_inputs

    tokenized_trainset = trainset.map(train_preprocess_function, batched=True, num_proc=1, remove_columns=trainset.column_names)

    if not ARGS.is_test:
        tokenized_trainset.save_to_disk(f'SummScreen/cached_tokenized/{fn}_train_cache')
        testset.save_to_disk(f'SummScreen/cached_tokenized/{fn}_test_cache')


def save_to(fname):
    save_dir = join('experiments', expname, 'checkpoints', fname)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

if ARGS.eval_first:
    rouges = ss.eval_epoch(0, testset)
    rouges_arr = np.array(rouges).mean(axis=0)
    print(f'Mean Rouge: {rouges_arr}')

alltime_best_rouges, all_rouges = ss.train_epochs(ARGS.n_epochs, tokenized_trainset, testset, ARGS.save_every, ARGS.eval_every)

def display_rouges(r):
    return list(zip(['r1','r2','rL','rLsum'],r))

print(display_rouges(alltime_best_rouges))

results_path = join('experiments',expname,'results.txt')
with open(results_path,'w') as f:
    for rname,rscore in display_rouges(alltime_best_rouges):
        f.write(f'{rname}: {rscore:.5f}\n')
    f.write('\nALL ROUGES:\n')
    for r in all_rouges:
        for rname, rscore in display_rouges(r):
            f.write(f'{rname}: {rscore:.5f}\t')
        f.write('\n')

summary_path = join('experiments',expname,'summary.txt')
with open(summary_path,'w') as f:
    f.write(f'Expname: {ARGS.expname}\n')
    f.write(f'captions: {ARGS.caps}\n')
    f.write(f'reorder: {ARGS.reorder}\n')
    f.write(f'randorder: {ARGS.randorder}\n')
    f.write(f'N Epochs: {ARGS.n_epochs}\n')
    f.write(f'Batch size: {ARGS.bs}\n')
