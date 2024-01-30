from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

pdset = load_dataset('json', data_files='SummScreen/baseline_valset.json')['train']
dset = load_dataset('json', data_files='SummScreen/baseline_testset.json')['train']

info_df = pd.read_csv('dset_info.csv', index_col=0)
icl_epnames = info_df.loc[info_df['usable']&(info_df['fragment']=='val')]
breakpoint()
for i in range(10):
    for summ_source in ('tvdb','tvmega_recap', 'soapcentral_condensed'):
        if pdset[summ_source][i] is not None and len()
    prompt += prompt_prefix+dset['transcript'][i]+prompt_suffix + pdset['tvdb']

prompt_prefix = 'Summarize the following TV show transcript.\n\n<Transcript Start>\n'
prompt_suffix = '\n<Transcript End>\n\nSummary:'
prompt = prompt_prefix + dset['transcript'][0] + prompt_suffix
model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

generated_ids = model.generate(**model_inputs, min_length=380, do_sample=True)
print(tokenizer.batch_decode(generated_ids)[0])
