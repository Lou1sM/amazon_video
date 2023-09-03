from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--cpu',action='store_true')
ARGS = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')
dset = load_dataset('json', data_files='SummScreen/scene_summs_to_summ.json')
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(dpoint):
    inputs = [doc for doc in dpoint['scene_summs']]
    model_inputs = tokenizer(inputs)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=dpoint['summ'])

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_dset_train = dset['train'].map(preprocess_function, batched=True, num_proc=1, remove_columns=dset['train'].column_names)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
dc = DataCollatorForSeq2Seq(tokenizer, model=model)

train_loader = DataLoader(tokenized_dset_train, batch_size=16, shuffle=True, collate_fn=dc)
opt = AdamW(model.parameters())

for epoch in range(3):
    for batch in train_loader:
        opt.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids[:,:64], attention_mask=attention_mask[:,:64], labels=labels[:,:64].contiguous())
        loss = outputs[0]
        loss.backward()
        print(loss)
        opt.step()

model.eval()
