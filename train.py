from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from dl_utils.misc import set_experiment_dir
from tqdm import tqdm
import numpy as np
from nelly_rouge import nelly_rouge
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
import argparse
import torch
from rouge_score import rouge_scorer
import os
from os.path import join
from transformers import get_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--retokenize',action='store_true')

parser.add_argument('--save_every',action='store_true')
parser.add_argument('--n_epochs',type=int,default=2)
parser.add_argument('--n_iter',type=int,default=-1)
parser.add_argument('--eval_every',type=int,default=1)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--reload_from',type=str,default='none')
parser.add_argument('--expname',type=str,default='tmp')
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn')
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('-bt','--is_test_big_bart',action='store_true')
parser.add_argument('--overwrite',action='store_true')
ARGS = parser.parse_args()

expname = set_experiment_dir(ARGS.expname, ARGS.overwrite, name_of_trials='tmp')
ARGS.is_test = ARGS.is_test or ARGS.is_test_big_bart
device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

model_name = 'lucadiliello/bart-small' if ARGS.is_test and not ARGS.is_test_big_bart else ARGS.model_name

print(f'using model {model_name}')

if ARGS.reload_from=='none':
    chkpt_path = model_name
else:
    chkpt_path = f'./experiments/{ARGS.reload_from}/chkpt-final'

print(f'loading from {chkpt_path}')
tokenizer = AutoTokenizer.from_pretrained(chkpt_path)

def old_get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return [raw_rscores[x].fmeasure for x in ('rouge1', 'rouge2', 'rougeLsum')]


if os.path.exists('cached_trainset') and os.path.exists('cached_testset') and not ARGS.retokenize:
    print('tokenized datasets have been cached, loading')
    tokenized_trainset = load_from_disk('cached_trainset')
    tokenized_testset = load_from_disk('cached_testset')
else:
    trainset = load_dataset('json', data_files='SummScreen/scene_summs_to_summ_train.json',split='train')
    testset = load_dataset('json', data_files='SummScreen/scene_summs_to_summ_test.json',split='train')

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

    tokenized_testset = testset.map(test_preprocess_function, batched=False, num_proc=1)
    tokenized_trainset = trainset.map(train_preprocess_function, batched=True, num_proc=1, remove_columns=trainset.column_names)

    tokenized_trainset.save_to_disk('cached_trainset')
    tokenized_testset.save_to_disk('cached_testset')


print(f'loading model from {chkpt_path}')
model = AutoModelForSeq2SeqLM.from_pretrained(chkpt_path).to(device)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)

train_loader = DataLoader(tokenized_trainset, batch_size=ARGS.batch_size, shuffle=False, collate_fn=dc)
def safe_decode(tokens):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def save_to(fname):
    save_dir = join('experiments', expname, 'checkpoints', fname)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

opt = AdamW(model.parameters(),lr=5e-5)
num_epochs = 3
num_training_steps = ARGS.n_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=opt, num_warmup_steps=0, num_training_steps=num_training_steps
    )
for epoch in range(ARGS.n_epochs):
    model.train()
    epoch_loss = 0
    rouges = []
    alltime_best_rouges = np.zeros(3)
    patience = 0
    print(f'training epoch {epoch}')
    for i,batch in enumerate(pbar := tqdm(train_loader, dynamic_ncols=True, smoothing=0.01, leave=False)):
        opt.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        cbatch = {k:v.cuda()[:,:tokenizer.model_max_length] for k,v in batch.items()}
        cbatch['labels'] = cbatch['labels'].contiguous()
        outputs = model(**cbatch)
        loss = outputs[0]
        loss.backward()
        epoch_loss = ((i*epoch_loss) + loss) / (i+1) # running avg
        pbar.set_description(f'Epoch: {epoch}/{ARGS.n_epochs}'
                             f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
        opt.step(); lr_scheduler.step()
        if i==ARGS.n_iter or (i==10 and ARGS.is_test):
            break
    if ARGS.save_every:
        save_to(f'epoch{i}')
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    print(f'Epoch: {epoch}\tLoss: {epoch_loss.item():.5f}')
    if (epoch+1) % ARGS.eval_every == 0:
        model.eval()
        print('validating')
        prev = ''
        epoch_rouge = np.zeros(3)
        for j,batch in enumerate(val_pbar := tqdm(tokenized_testset, dynamic_ncols=True, smoothing=0.01, leave=False)):
            tensor_inputs = torch.tensor(batch['input_ids'][:tokenizer.model_max_length],device=device)
            outputs = model.generate(tensor_inputs.unsqueeze(0),min_length=250,max_length=300)
            nl_outputs = safe_decode(outputs)[0]
            if nl_outputs[:100] == prev[:100]:
                breakpoint()
            prev = nl_outputs
            best_r2 = 0
            new_rouge = 0
            for k,possible_gt in batch.items():
                if k in ('input_ids','attention_mask') or possible_gt is None:
                    continue
                new_rouge = nelly_rouge(nl_outputs,possible_gt)
                if new_rouge[1] > best_r2:
                    best_r2 = new_rouge[1]
                    best_rouge = new_rouge
            if best_r2 == 0:
                if not all([v is None for k,v in batch.items() if k not in ('input_ids','attention_mask','ep_name')]):
                    breakpoint()
            rouges.append(best_rouge)
            epoch_rouge = ((j*epoch_rouge) + best_rouge) / (j+1) # running avg
            val_pbar.set_description(f'Epoch: {epoch}/{ARGS.n_epochs}'
                             f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f}  '
                             f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f}')
            if j==2 and ARGS.is_test:
                break

        rouges_arr = np.array(rouges).mean(axis=0)
        print(f'Mean Rouge: {rouges_arr}')
        if rouges_arr[1] > alltime_best_rouges[1]:
            patience = 0
            alltime_best_rouges = rouges_arr
            save_to('best')
        else:
            patience += 1
        if patience == 3:
            break

results_path = join('experiments',expname,'results.txt')
with open(results_path,'w') as f:
    for r,s in zip(['r1','r2','rL'],alltime_best_rouges):
        f.write(f'{r}: {s}\n')
