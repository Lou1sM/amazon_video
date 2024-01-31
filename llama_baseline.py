from tqdm import tqdm
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel, PeftConfig
from dl_utils.misc import check_dir
from dl_utils.tensor_funcs import cudify
from utils import rouge_from_multiple_refs
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_scheduler
from dl_utils.misc import set_experiment_dir
from datasets import load_dataset, load_from_disk
import argparse
import os
from os.path import join
import sys
from utils import display_rouges


#tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b') # always load this even if is_test
#tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-v0.1-GGUF') # always load this even if is_test
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1') # always load this even if is_test

prompt_prefix = 'Summarize the following TV show transcript.\n\n<Transcript Start>\n'
prompt_suffix = '\n<Transcript End>\n\nSummary:'
tok_pp = tokenizer(prompt_prefix)['input_ids']
tok_ps = tokenizer(prompt_suffix)['input_ids'][1:]

def load_peft_model(base_model_name_or_path, chkpt_path):
    print('loading model from', base_model_name_or_path)
    #model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path)
    model.cuda()
    return model
    if chkpt_path is None:
        print('no peft chkpt to update from')
        model = prepare_model_for_int8_training(model)
        lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model,lora_config)
    else:
        print('updating model with peft chkpt from', chkpt_path)
        config = PeftConfig.from_pretrained(chkpt_path)
        assert config.base_model_name_or_path==base_model_name_or_path
        model.enable_input_require_grads()
        model = PeftModel.from_pretrained(model, chkpt_path, is_trainable=True)
        assert any([x.requires_grad for x in model.parameters()])
    return model

def get_clipped(inputs, labs):
    max_len = min(1048, tokenizer.model_max_length)
    # add [2] (eos_token_id) manually because not added by tokenizer for some reason
    clipped_inputs = [tok_pp+x[1:max_len-len(lab+tok_pp+tok_ps)]+tok_ps+lab[1:] for x,lab in zip(inputs['input_ids'],labs['input_ids']) if len(lab+tok_pp+tok_ps)<max_len]
    clipped_labs = [[-100]*(min(len(x),max_len-len(lab+tok_pp+tok_ps))+len(tok_pp+tok_ps)-1)+lab[1:] for x,lab in zip(inputs['input_ids'],labs['input_ids']) if len(lab+tok_pp+tok_ps)<max_len]
    clipped_attn_masks = [[1]*len(x) for x in clipped_inputs]
    for cinp, clab, x, lab in zip(clipped_inputs, clipped_labs, inputs['input_ids'], labs['input_ids']):
        if len(cinp)!=len(clab):
            print(len(cinp), len(clab), len(x), len(lab))
            breakpoint()
    return clipped_inputs, clipped_labs, clipped_attn_masks

def train_preproc_fn(examples):
    inputs = tokenizer([dpoint for dpoint in examples['input']])
    assert all([x==1 for dp in inputs['attention_mask'] for x in dp])
    labels = tokenizer([dpoint+tokenizer.eos_token for dpoint in examples['output']])
    assert all([x==1 for dp in labels['attention_mask'] for x in dp])
    clipped_inputs, clipped_labels, clipped_attn_masks = get_clipped(inputs, labels)
    clipped_attn_masks = [[1]*len(x) for x in clipped_inputs]
    results = {}
    results['input_ids'] = clipped_inputs
    results['labels'] = clipped_labels
    results['attention_mask'] = clipped_attn_masks

    assert all(len(x)==len(y) for x,y in zip(results['input_ids'],results['labels']))
    assert all(len(x)==len(y) for x,y in zip(results['labels'],results['attention_mask']))
    return results

def first_n_tokens(text, n):
    ts = text.split()
    go_up_to_idx = len(' '.join(ts[:n*3//4]))
    return text[:go_up_to_idx] #idx orig to keep whitespace right

def test_preproc_fn(examples):
    inputs = tokenizer([prompt_prefix + first_n_tokens(dpoint,1240) + prompt_suffix for dpoint in examples['transcript']])['input_ids']
    attn_masks = [[1]*len(x) for x in inputs]
    results = {}
    results['input_ids'] = inputs
    results['attention_mask'] = attn_masks
    for k in ('epname', 'soapcentral_condensed', 'tvmega_recap', 'imdb'):
        results[k] = examples[k]

    assert all(len(x)==len(y) for x,y in zip(results['input_ids'],results['attention_mask']))
    return results

def get_maybe_cached_dset(fragment, preproc_fn):
    cache_path = f'SummScreen/cached_tokenized/{fragment}set_for_llama_baseline'
    if os.path.exists(cache_path) and not ARGS.retokenize:
        print(f'dataset has been cached on disk at {cache_path}, loading from there')
        tokenized_dset = load_from_disk(cache_path)
    else:
        json_dset_path = f'SummScreen/baseline_{fragment}set.json'
        print(f'no cached data found on disk at {cache_path}, loading json datset from {json_dset_path}')
        dset = load_dataset('json', data_files=json_dset_path)
        tokenized_dset = dset['train'].map(preproc_fn, batched=True, num_proc=1, remove_columns=dset['train'].column_names)
        tokenized_dset.save_to_disk(cache_path)
    return tokenized_dset

parser = argparse.ArgumentParser()
parser.add_argument('--bs',type=int,default=1)
parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions','kosmos-only','swinbert-only'], default='nocaptions')
parser.add_argument('--expname',type=str)
parser.add_argument('--expdir_prefix',type=str,default='experiments')
parser.add_argument('--n_dpoints',type=int,default=-1)
parser.add_argument('--n_epochs',type=int,default=10)
parser.add_argument('--reload_from',type=str, default=None)
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--retokenize',action='store_true')
ARGS = parser.parse_args()

if ARGS.expname is None and not ARGS.is_test:
    sys.exit('must set explicit expname when not in test mode')
elif ARGS.is_test:
    ARGS.expname='llamatmp'
    ARGS.n_epochs = min(ARGS.n_epochs,2)
    ARGS.n_dpoints = 10
if not ARGS.expname.startswith('llama'):
    ARGS.expname = 'llama'+ARGS.expname

expdir = join(ARGS.expdir_prefix,ARGS.expname)
#base_model_name = 'seanmor5/tiny-llama-test' if ARGS.is_test else 'huggyllama/llama-7b'
#base_model_name = 'mistralai/Mistral-7B-v0.1'
base_model_name = 'mistralai/Mistral-7B-v0.1'
reload_chkpt_path = None if ARGS.reload_from is None else f'{expdir}/checkpoints/{ARGS.reload_from}'
if ARGS.reload_from is None:
    set_experiment_dir(expdir, ARGS.overwrite, name_of_trials=join(ARGS.expdir_prefix,'llamatmp'))
else:
    assert os.path.exists(expdir)

tokenized_trainset = get_maybe_cached_dset('train', train_preproc_fn)
tokenized_valset = get_maybe_cached_dset('val', test_preproc_fn)
tokenized_testset = get_maybe_cached_dset('test', test_preproc_fn)

model = load_peft_model(base_model_name, reload_chkpt_path)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)
trainloader = DataLoader(tokenized_trainset, batch_size=ARGS.bs, shuffle=True, collate_fn=dc)
tokenizer.pad_token = tokenizer.eos_token

def inference_epoch(dset,fragment):
    rouges = []
    prev = ''
    epoch_rouge = np.zeros(4)
    check_dir(generations_dir := join(expdir, f'generations_{fragment}'))
    pbar = tqdm(range(0,len(dset),ARGS.bs))
    model.eval()
    for j in pbar:
        batch = dset[j*ARGS.bs:(j+1)*ARGS.bs]
        pad_len = max(len(x) for x in batch['input_ids'])
        padded_inputs = [x+[tokenizer.pad_token_id]*(pad_len-len(x)) for x in batch['input_ids']]
        padded_inputs = cudify(padded_inputs)
        with torch.no_grad():
            preds = model.generate(input_ids=padded_inputs, min_length=2048, max_length=2048)
        nl_outputs = tokenizer.batch_decode([p[len(binp):]  for p,binp in zip(preds,batch['input_ids'])], skip_special_tokens=True, cleanup_tokenization_spaces=True)
        assert len(nl_outputs) == ARGS.bs
        #nl_outputs = nl_outputs_[0]
        if (nl_outputs[:100] == prev[:100]):# and not (prev_inp[:100] == batch['input_ids'][:100]):
            print('repeat output')
        prev = nl_outputs
        for i, nlo in enumerate(nl_outputs):
            print(nlo)
            references = [v[i] for k,v in batch.items() if k not in ('input_ids','attention_mask') and v[i] is not None]
            best_rouge = rouge_from_multiple_refs(nlo, references, return_full=False, benchmark_rl=True)
            rouges.append(best_rouge)
        epoch_rouge = (((j+i)*epoch_rouge) + best_rouge) / (j+i+1) # running avg
        pbar.set_description(f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f} {best_rouge[3]:.3f}  '
                         f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f} {epoch_rouge[3]:.3f}')
        epname = batch['epname']
        for en, nlo in zip(batch['epname'], nl_outputs):
            with open(f'{generations_dir}/{en}.txt','w') as f:
                f.write(nlo)
        if (j==2 and ARGS.is_test) or (j==ARGS.n_dpoints-1):
            break
    return np.array(rouges)

to_opt = model.parameters() if ARGS.is_test else model.model.model.layers[24:].parameters()
opt = AdamW(model.parameters(),lr=1e-6)
for p in model.parameters():
    if p not in to_opt:
        p.requires_grad=False

num_training_steps = ARGS.n_epochs * len(trainloader)
lr_scheduler = get_scheduler(name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps)

def save_model(save_path):
    print(f'saving to {save_path}')
    model.save_pretrained(save_path)
best_val_rouges = np.zeros(4)
patience = 0
epoch_loss = 0
best_chkpt_path = f'{expdir}/checkpoints/best'
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
        if ARGS.is_test or (ARGS.n_dpoints!=-1 and i*ARGS.bs >= ARGS.n_dpoints):
            break
    save_model(f'{expdir}/checkpoints/epoch{en}')
    save_model(f'{expdir}/checkpoints/best')
    #val_rouges = inference_epoch(tokenized_valset, 'val').mean(axis=0)
    #if val_rouges[2] > best_val_rouges[2]:
    #    best_val_rouges = val_rouges
    #    save_model(f'{expdir}/checkpoints/best')
    #else:
    #    patience += 1
    #print(f'Mean Rouge: {val_rouges}\tPatience: {patience}\t')
    if patience == 2:
        break

#if os.path.exists(best_chkpt_path):
    #model = load_peft_model(base_model_name, best_chkpt_path)
#else:
    #assert best_val_rouges[2]==0 # should only happen if all rouges remained zero for some reason
test_rouges = inference_epoch(tokenized_testset, 'test').mean(axis=0)
results_path = join(expdir,'results.txt')
with open(results_path,'w') as f:
    f.write('\nTEST ROUGES:\n')
    for rname,rscore in display_rouges(test_rouges):
        f.write(f'{rname}: {rscore:.5f}\n')
