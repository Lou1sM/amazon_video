import re
import os
from dl_utils.tensor_funcs import numpyify, cudify
import rouge
import torch
import numpy as np
from natsort import natsorted
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig


rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                                 max_n=2,
                                                 limit_length=False,
                                                 apply_avg=True,
                                                 apply_best=False,
                                                 alpha=0.5, # Default F1_score
                                                 stemming=False)

def display_rouges(r):
    return list(zip(['r1','r2','rL','rLsum'],r))

def rouge_preprocess(text):
    text = text.replace('...',' <eplipsis> ')
    text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
    tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
    rouge.Rouge.stem_tokens(tokens)
    preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
    return preprocessed_text

def nelly_rouge(pred_,gt_):
    pred_sum_sents = [rouge_preprocess(p) for p in split_summ(pred_)]
    gt_sum_sents = [rouge_preprocess(g) for g in split_summ(gt_)]
    pred = '\n'.join(pred_sum_sents)
    gt = '\n'.join(gt_sum_sents)

    scores = rouge_eval.get_scores(pred, gt)

    pred_old = [rouge_preprocess(pred_)]
    gt_old = [rouge_preprocess(gt_)]
    old_scores = rouge_eval.get_scores(pred_old, gt_old)
    scores['rouge-lsum'] = scores['rouge-l']
    scores['rouge-l'] = old_scores['rouge-l']
    return scores

def old_nelly_rouge(pred,gt):
    if not isinstance(pred,list):
        pred = [pred]
    if not isinstance(gt,list):
        gt = [gt]
    pred_sums = [rouge_preprocess(pred) for pred in pred]
    gt_sums = [rouge_preprocess(g) for g in gt]
    scores = rouge_eval.get_scores(pred_sums, gt_sums)
    return scores

def split_summ(s):
    return s.replace('. ','.\n').split('\n')

def extract_main_rouges(scores):
    rouge1 = scores['rouge-1']['f'] * 100
    rouge2 = scores['rouge-2']['f'] * 100
    rougel = scores['rouge-l']['f'] * 100
    rougelsum = scores['rouge-lsum']['f'] * 100
    return rouge1, rouge2, rougel, rougelsum

def rouge_from_multiple_refs(pred, references, return_full, benchmark_rl):
    benchmark = -1
    for possible_gt in references:
        new_rouge = nelly_rouge(pred, possible_gt)
        maybe_new_benchmark = new_rouge['rouge-l']['f'] if benchmark_rl else new_rouge['rouge-2']['f']
        if maybe_new_benchmark > benchmark:
            benchmark = maybe_new_benchmark
            best_rouge = new_rouge
    if benchmark == 0:
        if not all([gt is None for gt in references]):
            print('rouge is zero')
    return best_rouge if return_full else extract_main_rouges(best_rouge)

def get_fn(caps, order, uniform_breaks, startendscenes, centralscenes, is_test):
    fn = caps
    if order=='optimal':
        fn += '_reordered'
    if order=='rand':
        fn += '_rand_ordered'
    if uniform_breaks:
        fn += '_uniform_breaks'
    if startendscenes:
        fn += '_startendscenes'
    if centralscenes:
        fn += '_centralscenes'
    if is_test:
        fn += '_test'
    #if n_dpoints != -1:
        #fn += f'{n_dpoints}dps'
    return fn

def safe_decode(tokens, tokenizer):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def chunkify(text,max_chunk_size):
    if len(text.split())*4/3 < max_chunk_size:
        to_return = [text]
    else:
        first_chunk, second_chunk = split_text_by_sth(text)
        to_return = chunkify(first_chunk,max_chunk_size) + chunkify(second_chunk,max_chunk_size)
    if not all(len(x) <= max_chunk_size for sl in to_return for x in sl):
        breakpoint()
    return to_return

def split_text_by_sth(text):
    for sep in ('\n', '. ', ', ', ' '):
        if sep in text.strip():
            return split_text_by_sep(text.strip(),sep)
    return text[:len(text)//2], text[len(text)//2:]

def summ_short_scene(text):
    return ' '.join(convert_script_to_prose(line) for line in text.split('\n') if line!='')

def convert_script_to_prose(script_line):
    if maybe_speaker_name:=re.match(r'\w+: ', script_line):
        speaker_name = script_line[:maybe_speaker_name.span()[1]-2]
        speech = script_line[maybe_speaker_name.span()[1]:]
        return f'{speaker_name} said "{speech}"'
    elif stage_direction := re.match(r'(?<=\[ )[A-Z -]+(?= \])', script_line):
        return stage_direction
    else:
        return script_line

def split_text_by_sep(text,sep):
    lines = text.split(sep)
    N = len(text.split())
    first_chunk = ''
    for i,l in enumerate(lines):
        if abs(len((first_chunk+l).split()) - N/2) > abs(len(first_chunk.split())-N/2):
            break # get as close to halfway as possible
        if first_chunk=='':
            first_chunk = l+sep
        else:
            first_chunk += l+sep
        if not text.startswith(first_chunk):
            breakpoint()
    second_chunk = text[len(first_chunk):]
    assert first_chunk+second_chunk == text
    return first_chunk, second_chunk

def prepare_for_pil(torch_im):
    assert isinstance(torch_im, torch.Tensor)
    normed_torch_im = (torch_im - torch_im.min()) / (torch_im.max() - torch_im.min())
    np_im = normed_torch_im.permute(1,2,0).numpy()
    np_uint8_im = (np_im*255).astype(np.uint8)
    return np_uint8_im

def shim(im):
    import matplotlib.pyplot as plt
    plt.imshow(im); plt.show()

def tshim(t):
    a = numpyify(t.permute(1,2,0))
    shim(a)

def path_list(parent_dir):
    return natsorted([os.path.join(parent_dir, child) for child in os.listdir(parent_dir)])

def bernoulli_CE(p1, p2):
    return -p1 * np.log(p2) - (1-p1)*np.log(1-p2)

def load_peft_model(base_model_name_or_path, chkpt_path):
    print('loading model from', base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, load_in_8bit=True)
    if chkpt_path is None:
        print('no peft chkpt to update from')
        model = prepare_model_for_kbit_training(model)
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

