from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from dl_utils.misc import check_dir
from dl_utils.tensor_funcs import cudify
from utils import rouge_from_multiple_refs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from factscore.utils import convert_model_to_int8_on_gpu
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_scheduler
from dl_utils.misc import set_experiment_dir
from datasets import load_dataset, load_from_disk
import argparse
import torch
import os
from os.path import join
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--bs',type=int,default=1)
parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions','kosmos-only','swinbert-only'], default='nocaptions')
parser.add_argument('--expname',type=str)
parser.add_argument('--expdir_prefix',type=str,default='experiments')
parser.add_argument('--n_dpoints',type=int,default=-1)
parser.add_argument('--n_epochs',type=int,default=10)
parser.add_argument('--reload_from',type=str, default='none')
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--retokenize',action='store_true')
ARGS = parser.parse_args()

if ARGS.expname is None and not ARGS.is_test:
    sys.exit('must set explicit expname when not in test mode')
elif ARGS.is_test:
    ARGS.expname='llamatmp'
    ARGS.n_epochs = 2
    ARGS.n_dpoints = 10
if not ARGS.expname.startswith('llama'):
    ARGS.expname = 'llama'+ARGS.expname

expdir = set_experiment_dir(join(ARGS.expdir_prefix,ARGS.expname), ARGS.overwrite, name_of_trials=join(ARGS.expdir_prefix,'llamatmp'))
if ARGS.reload_from=='none':
    chkpt_path = 'huggyllama/llama-13b'
else:
    chkpt_path = f'./{expdir}/checkpoints/epoch{ARGS.reload_from}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(chkpt_path)

def train_preproc_fn(examples):
    inputs = tokenizer(['Summarize the following TV show:' + dpoint for dpoint in examples['input']])
    assert all([x==1 for dp in inputs['attention_mask'] for x in dp])
    labels = tokenizer([dpoint for dpoint in examples['output']])
    assert all([x==1 for dp in labels['attention_mask'] for x in dp])
    clipped_inputs = [x[:tokenizer.model_max_length-len(lab)]+lab for x,lab in zip(inputs['input_ids'],labels['input_ids']) if len(lab)<tokenizer.model_max_length]
    clipped_labels = [[-100]*(min(len(x),tokenizer.model_max_length-len(lab)))+lab for x,lab in zip(inputs['input_ids'],labels['input_ids']) if len(lab)<tokenizer.model_max_length]
    clipped_attn_masks = [[1]*len(x) for x in clipped_inputs]
    results = {}
    results['input_ids'] = clipped_inputs
    results['labels'] = clipped_labels
    results['attention_mask'] = clipped_attn_masks

    assert all(len(x)==len(y) for x,y in zip(results['input_ids'],results['labels']))
    assert all(len(x)==len(y) for x,y in zip(results['labels'],results['attention_mask']))
    return results

def test_preproc_fn(examples):
    inputs = tokenizer(['Summarize the following TV show:' + dpoint for dpoint in examples['transcript']])
    assert all([x==1 for dp in inputs['attention_mask'] for x in dp])
    clipped_inputs = [x[:tokenizer.model_max_length] for x in inputs['input_ids']]
    clipped_attn_masks = [[1]*len(x) for x in clipped_inputs]
    results = {}
    results['input_ids'] = clipped_inputs
    results['attention_mask'] = clipped_attn_masks
    for k in ('epname', 'soapcentral_condensed', 'tvmega_recap', 'imdb'):
        results[k] = examples[k]

    assert all(len(x)==len(y) for x,y in zip(results['input_ids'],results['attention_mask']))
    return results

def get_maybe_cached_dset(fragment, preproc_fn):
    cache_path = f'SummScreen/cached_tokenized/{fragment}set_for_baseline'
    if os.path.exists(cache_path) and not ARGS.retokenize:
        tokenized_dset = load_from_disk(cache_path)
    else:
        dset = load_dataset('json', data_files=f'SummScreen/baseline_{fragment}set.json')
        tokenized_dset = dset['train'].map(preproc_fn, batched=True, num_proc=1, remove_columns=dset['train'].column_names)
        tokenized_dset.save_to_disk(cache_path)
    return tokenized_dset

tokenized_trainset = get_maybe_cached_dset('train', train_preproc_fn)
tokenized_valset = get_maybe_cached_dset('val', test_preproc_fn)
tokenized_testset = get_maybe_cached_dset('test', test_preproc_fn)

model = AutoModelForCausalLM.from_pretrained('seanmor5/tiny-llama-test' if ARGS.is_test else chkpt_path, load_in_8bit=True)
model = prepare_model_for_int8_training(model)

lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model,lora_config)
#model = convert_model_to_int8_on_gpu(model, device='cuda')

dc = DataCollatorForSeq2Seq(tokenizer, model=model)
trainloader = DataLoader(tokenized_trainset, batch_size=ARGS.bs, shuffle=True, collate_fn=dc)
tokenizer.pad_token = tokenizer.eos_token
#trainloader = DataLoader(tokenized_trainset, batch_size=4, shuffle=True)

def inference_epoch(dset,fragment):
    rouges = []
    prev = ''
    epoch_rouge = np.zeros(4)
    check_dir(generations_dir := join(expdir, f'generations_{fragment}'))
    for j,batch in enumerate(pbar := tqdm(dset, dynamic_ncols=True, smoothing=0.01, leave=False)):
        preds = model.generate(input_ids=cudify([batch['input_ids']]),attention_mask=cudify([batch['attention_mask']]))
        nl_outputs_ = tokenizer.batch_decode(preds)
        assert len(nl_outputs_) == 1
        nl_outputs = nl_outputs_[0]
        if (nl_outputs[:100] == prev[:100]):# and not (prev_inp[:100] == batch['input_ids'][:100]):
            print('repeat output')
        prev = nl_outputs
        references = [v for k,v in batch.items() if k not in ('input_ids','attention_mask') and v is not None]
        best_rouge = rouge_from_multiple_refs(nl_outputs, references, return_full=False, benchmark_rl=True)

        rouges.append(best_rouge)
        epoch_rouge = ((j*epoch_rouge) + best_rouge) / (j+1) # running avg
        pbar.set_description(f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f} {best_rouge[3]:.3f}  '
                         f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f} {epoch_rouge[3]:.3f}')
        epname = batch['epname']
        with open(f'{generations_dir}/{epname}.txt','w') as f:
            f.write(nl_outputs)
        if j==2 and ARGS.is_test:
            break
    return np.array(rouges)

to_opt = model.parameters() if ARGS.is_test else model.model.model.layers[32:].parameters()
for p in model.model.model.layers[:32].parameters():
    p.require_grad = False

opt = AdamW(to_opt,lr=1e-6)
num_training_steps = ARGS.n_epochs * len(trainloader)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

best_val_rl = 0
patience = 0
epoch_loss = 0
for en in range(ARGS.n_epochs):
    print('Epoch',en)
    model.train()
    opt.zero_grad()
    for i,batch in enumerate(pbar:=tqdm(trainloader)):
        opt.zero_grad()
        cbatch = {k:cudify(v) for k,v in batch.items()}
        cbatch['labels'] = cbatch['labels'].contiguous()
        outputs = model(**cbatch)
        loss = outputs[0]
        loss.backward()
        opt.step(); lr_scheduler.step()
        epoch_loss = ((i*epoch_loss) + loss.item()) / (i+1) # running avg
        pbar.set_description(f'Epoch: {en}/{ARGS.n_epochs}'
                                 f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
        if ARGS.is_test:
            break
    val_rouges = inference_epoch(tokenized_valset, 'val').mean(axis=0)
    if val_rouges[2] > best_val_rl:
        best_val_rl = val_rouges[2]
    else:
        patience += 1
    print(f'Mean Rouge: {val_rouges}\tPatience: {patience}')
    if patience == 2:
        break
test_rouges = inference_epoch(tokenized_testset, 'test').mean(axis=0)