from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np
from nelly_rouge import nelly_rouge
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
import argparse
import torch
from rouge_score import rouge_scorer


parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
parser.add_argument('--n_epochs',type=int,default=50)
parser.add_argument('--eval_every',type=int,default=10)
parser.add_argument('--model_name',type=str,default='facebook/bart-large-cnn')
ARGS = parser.parse_args()


def old_get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return [raw_rscores[x].fmeasure for x in ('rouge1', 'rouge2', 'rougeLsum')]

device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')
trainset = load_dataset('json', data_files='SummScreen/scene_summs_to_summ_train.json',split='train')
testset = load_dataset('json', data_files='SummScreen/scene_summs_to_summ_test.json',split='train')
tokenizer = AutoTokenizer.from_pretrained(ARGS.model_name)

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
        model_inputs[k] = [tokenizer.pad_token_id] if v is None else tokenizer(v)['input_ids']
    return model_inputs

model = AutoModelForSeq2SeqLM.from_pretrained(ARGS.model_name).to(device)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)

tokenized_testset = testset.map(test_preprocess_function, batched=False, num_proc=1)
test_loader = DataLoader(tokenized_testset, batch_size=1, shuffle=False, collate_fn=dc)

tokenized_trainset = trainset.map(train_preprocess_function, batched=True, num_proc=1, remove_columns=trainset.column_names)
train_loader = DataLoader(tokenized_trainset, batch_size=2, shuffle=True, collate_fn=dc)

def safe_decode(tokens):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

opt = AdamW(model.parameters(),lr=1e-4)

for epoch in range(ARGS.n_epochs):
    model.train()
    epoch_loss = 0
    model.eval()
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
        epoch_loss = ((i*epoch_loss) + loss) / (i+1) # running avg
        opt.step()
    if (epoch)%ARGS.eval_every == 0:
        rouges = []
        old_rouges = []
        for batch in test_loader:
            outputs = model.generate(batch['input_ids'][:,:tokenizer.model_max_length].cuda(),min_length=250,max_length=300)
            nl_inputs = safe_decode(labels)
            nl_outputs = safe_decode(outputs)
            rouge = nelly_rouge(nl_outputs,nl_inputs)
            old_rouge = [old_get_rouges(pred,gt) for pred,gt in zip(nl_outputs,nl_inputs)]
            old_rouges += old_rouge
            rouges.append(rouge)
        for r in (old_rouges, rouges):
            rouges_arr = np.array(r)
            print(f'Mean Rouge: {rouges_arr.mean(axis=0)}')
    print(f'Epoch: {epoch}\tLoss: {epoch_loss.item():.5f}')

    model_fname = ARGS.model_name.split('/')[-1]
    save_dir = f'checkpoints/finetuned-{model_fname}'
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

