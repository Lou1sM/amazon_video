from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
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
from transformers import get_scheduler


parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--retokenize',action='store_true')
parser.add_argument('--save_every',action='store_true')
parser.add_argument('--n_epochs',type=int,default=2)
parser.add_argument('--n_iter',type=int,default=-1)
parser.add_argument('--eval_every',type=int,default=1)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--finetuned',type=int,default=0)
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn') 
parser.add_argument('-t','--is_test',action='store_true')
ARGS = parser.parse_args()

model_name = 'lucadiliello/bart-small' if ARGS.is_test else ARGS.model_name

def old_get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return [raw_rscores[x].fmeasure for x in ('rouge1', 'rouge2', 'rougeLsum')]

device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)

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


if ARGS.finetuned==0:
    chkpt_path = model_name
else:
    chkpt_path = f'./finetuned/{model_name}/chkpt{ARGS.finetuned}'


print(f'loading model from {chkpt_path}')
model = AutoModelForSeq2SeqLM.from_pretrained(chkpt_path).to(device)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)

train_loader = DataLoader(tokenized_trainset, batch_size=ARGS.batch_size, shuffle=False, collate_fn=dc)
#test_loader = DataLoader(tokenized_testset, batch_size=1, shuffle=False, collate_fn=dc)
def safe_decode(tokens):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

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
    alltime_best_r2 = 0
    patience = 0
    print(f'training epoch {epoch}')
    for j,batch in enumerate(pbar := tqdm(train_loader, dynamic_ncols=True, smoothing=0.01, leave=False)):
        opt.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        cbatch = {k:v.cuda()[:,:tokenizer.model_max_length] for k,v in batch.items()}
        cbatch['labels'] = cbatch['labels'].contiguous()
        outputs = model(**cbatch)
        #if (j+1)%100 == 0:
        #if epoch == ARGS.eval_every-1:
        #    n = min(250,len(cbatch['labels']))
        #    pred_ids = model.generate(cbatch['input_ids'].cuda(),min_length=n,max_length=300)
        #    preds = safe_decode(pred_ids)[0]
        #    print(preds)
        #    gt = safe_decode(batch['labels'])[0]
        #    n = min(len(gt),len(preds))
        #    if gt[:n]==preds[:n]:
        #        print('PERFECT')
        #    else:
        #        print(gt)
        #    if 'RECAP' in gt or 'RECAP' in preds:
        #        breakpoint()
        loss = outputs[0]
        loss.backward()
        ema = loss if j==0 else (ema+loss)/2 # EMA moving avg
        epoch_loss = ((j*epoch_loss) + loss) / (j+1) # running avg
        pbar.set_description(f'Epoch: {epoch}/{ARGS.n_epochs}'
                             f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
        opt.step(); lr_scheduler.step()
        if j==ARGS.n_iter or (j==10 and ARGS.is_test):
            break
    if ARGS.save_every:
        save_dir = f'./finetuned_checkpoints/{model_name}/chkpt{epoch}'
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    print(f'Epoch: {epoch}\tLoss: {epoch_loss.item():.5f}')
    if (epoch+1) % ARGS.eval_every == 0:
        model.eval()
        print('validating')
        prev = ''
        i = 0
        for batch in tqdm(tokenized_testset):
            tensor_inputs = torch.tensor(batch['input_ids'][:tokenizer.model_max_length],device=device)
            outputs = model.generate(tensor_inputs.unsqueeze(0),min_length=250,max_length=300)
            nl_outputs = safe_decode(outputs)[0]
            #print(nl_outputs)
            if nl_outputs[:100] == prev[:100]:
                breakpoint()
            prev = nl_outputs
            best_r2 = 0
            new_rouge = 0
            for k,possible_gt in batch.items():
                if k in ('input_ids','attention_mask') or possible_gt is None:
                    continue
                #print(possible_gt)
                new_rouge = nelly_rouge(nl_outputs,possible_gt)
                if new_rouge[1] > best_r2:
                    best_r2 = new_rouge[1]
                    best_rouge = new_rouge
            if best_r2 == 0:
                if not all([v is None for k,v in batch.items() if k not in ('input_ids','attention_mask','ep_name')]):
                    breakpoint()
            rouges.append(best_rouge)
            i+=1
            if i==3 and ARGS.is_test:
                break

        rouges_arr = np.array(rouges).mean(axis=0)
        print(f'Mean Rouge: {rouges_arr}')
        if rouges_arr[1] > alltime_best_r2:
            patience = 0
            alltime_best_r2 = rouges_arr[1]
        else:
            patience += 1
        if patience == 3:
            break

save_dir = f'./finetuned_checkpoints/{model_name}/chkpt-final'
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

