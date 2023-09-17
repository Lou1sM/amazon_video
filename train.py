from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np
from nelly_rouge import nelly_rouge
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
import argparse
import torch
from rouge_score import rouge_scorer
import os


parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--retokenize',action='store_true')
parser.add_argument('--n_epochs',type=int,default=50)
parser.add_argument('--eval_every',type=int,default=10)
parser.add_argument('--print_loss_every',type=int,default=10)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn')
ARGS = parser.parse_args()


def old_get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return [raw_rscores[x].fmeasure for x in ('rouge1', 'rouge2', 'rougeLsum')]

device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name)

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

model = AutoModelForSeq2SeqLM.from_pretrained(ARGS.model_name).to(device)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)

train_loader = DataLoader(tokenized_trainset, batch_size=ARGS.batch_size, shuffle=True, collate_fn=dc)
test_loader = DataLoader(tokenized_testset, batch_size=1, shuffle=False, collate_fn=dc)

def safe_decode(tokens):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

opt = AdamW(model.parameters(),lr=1e-4)

for epoch in range(ARGS.n_epochs):
    model.train()
    epoch_loss = 0
    model.eval()
    j = 0
    if (epoch)%ARGS.eval_every == 0:
        rouges = []
        old_rouges = []
        for batch in tokenized_testset:
            tensor_inputs = torch.tensor(batch['input_ids'][:tokenizer.model_max_length],device=device)
            outputs = model.generate(tensor_inputs.unsqueeze(0),min_length=250,max_length=300)
            nl_outputs = safe_decode(outputs)[0]
            best_r2 = 0
            new_rouge = 0
            for k,v in batch.items():
                if k in ('input_ids','attention_mask') or v is None:
                    continue
                #possible_gt = safe_decode(v)
                possible_gt = v
                new_rouge = nelly_rouge(nl_outputs,possible_gt)
                if new_rouge[1] > best_r2:
                    best_r2 = new_rouge[1]
                    best_rouge = new_rouge
                    old_rouge = old_get_rouges(nl_outputs,possible_gt)
            if best_r2 == 0:
                breakpoint()
            old_rouges += old_rouge
            rouges.append(best_rouge)

        for r in (old_rouges, rouges):
            rouges_arr = np.array(r)
            print(f'Mean Rouge: {rouges_arr.mean(axis=0)}')
    for i, batch in enumerate(train_loader):
        opt.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        cbatch = {k:v.cuda()[:,:tokenizer.model_max_length] for k,v in batch.items()}
        cbatch['labels'] = cbatch['labels'].contiguous()
        outputs = model(**cbatch)
        loss = outputs[0]
        loss.backward()
        epoch_loss = ((j*epoch_loss) + loss) / (j+1) # running avg
        if j == ARGS.print_loss_every:
            print(f'Batch {j} loss: {loss.item():.5f}')
            j = 0
            epoch_loss = 0
        opt.step()
        j+=1
    print(f'Epoch: {epoch}\tLoss: {epoch_loss.item():.5f}')

    model_fname = ARGS.model_name.split('/')[-1]
    save_dir = f'checkpoints/finetuned-{model_fname}'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

