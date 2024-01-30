from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import torch
import pandas as pd


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", load_in_8bit=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

pdset = load_dataset('json', data_files='SummScreen/baseline_valset.json')['train']
dset = load_dataset('json', data_files='SummScreen/baseline_testset.json')['train']

info_df = pd.read_csv('dset_info.csv', index_col=0)
icl_epnames = info_df.loc[info_df['usable']&(info_df['split']=='val')]
prompt_prefix = 'Summarize the following TV show transcript.\n\n<Transcript Start>\n'
prompt_suffix = '\n<Transcript End>\n\nSummary:'
prompt = ''
for i in range(10):
    summ = 'none'
    for summ_source in ('tvdb','tvmega_recap', 'soapcentral_condensed'):
        potential_summ = pdset[summ_source][i]
        if potential_summ is not None and len(potential_summ.split()) > 100:
            summ = potential_summ
            break
    assert summ != 'none'
    prompt += prompt_prefix+pdset['transcript'][i]+prompt_suffix+summ

prompt += prompt_prefix + dset['transcript'][0] + prompt_suffix
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

print('generating')
with torch.no_grad():
    generated_ids = model.generate(**model_inputs, min_length=380, max_length=400, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
