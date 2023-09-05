from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from time import time
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode, episode_from_ep_name
#from torchmetrics.text.rouge import ROUGEScore
from rouge_score import rouge_scorer
import numpy as np


class SoapSummer():
    def __init__(self,device):
        self.device = device
        self.dtokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
        self.dmodel = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary").to(self.device)

        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        self.bs = 8
        self.dbs = 8

    def pad_batch(self,batch):
        N=max([len(c) for c in batch])
        padded = [b+[self.dtokenizer.eos_token_id]*(N-len(b)) for b in batch]
        return torch.tensor(padded).to(self.device)

    def summ_scenes(self,ep):
        start_time = time()
        chunk_list = [self.chunkify(s,level='dialogue') for s in ep.scenes]
        #scene_num_list = [f'SCENE{i}' for i,x in enumerate(chunk_list) for j in range(len(x))]
        chunks = sum(chunk_list,[])
        sort_idxs = np.argsort([len(x) for x in chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_chunks = [chunks[i] for i in sort_idxs]
        if not all([sorted_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(chunks)]):
            breakpoint()
        #padded_batch = self.pad_batch(chunks)
        N = ceil(len(chunks)/self.dbs)
        chunk_summs = []
        for i in range(N):
            padded = self.pad_batch(sorted_chunks[i*self.dbs:(i+1)*self.dbs])
            print(padded.shape)
            max_len = min(padded.shape[1],50)
            min_len = max(10,max_len-20)
            summ_tokens = self.dmodel.generate(padded,min_length=min_len,max_length=max_len)
            summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            #summ = [re.match(r'.*\<s\>(.*)\<\/s\>',s).groups()[0] for s in summ]
            chunk_summs += summ
        print(f'Scene summ time: {time()-start_time:.2f}')
        desorted_chunk_summs = [chunk_summs[i] for i in reversed_sort_idxs]
        # now reuinfy whatever scenes were split into chunks
        count = 0
        desplit = []
        for cl in chunk_list:
            desplit.append(' '.join(desorted_chunk_summs[count:count+len(cl)]))
            count+=len(cl)
        assert (desplit==desorted_chunk_summs) == (set([len(x) for x in chunk_list])==set([1]))
        with open(f'SummScreen/scene_summs/{ep.ep_name}.txt','w') as f:
            f.write('\n'.join(desplit))
        return desplit

    def get_scene_summs(self,ep_name):
        maybe_scene_summ_path = f'SummScreen/scene_summs/{ep.ep_name}.txt'
        if os.path.exists(maybe_scene_summ_path):
            with open(maybe_scene_summ_path) as f:
                return f.readlines()
        else:
            return self.summ_scenes(ep)

    def summarize(self,ep,recompute=False):
        start_time = time()
        scene_summs = self.get_scene_summs(ep.ep_name)
        concatted_scene_summs = '\n'.join(scene_summs)
        chunks = self.chunkify(concatted_scene_summs,level='meta')
        assert len(chunks) < self.bs
        #mean_gt_summ_len = sum([len(s.split()) for s in ep.summaries.values()])/len(ep.summaries) # soap_central v long, kinda skewing it
        #max_len = min(mean_gt_summ_len/len(chunks),50) # so just hard-code for now
        max_len = 300
        min_len = max(10,max_len-20)
        meta_chunk_summs = self.model.generate(self.pad_batch(chunks),min_length=min_len,max_length=max_len)
        final_summ = ' '.join(self.tokenizer.batch_decode(meta_chunk_summs,skip_special_tokens=True))
        print(f'Total summ time: {time()-start_time:.2f}')
        return final_summ

    def chunkify(self,text,level='dialogue'):
        if level == 'dialogue':
            tokenizer = self.dtokenizer
        else:
            assert level=='meta'
            tokenizer = self.tokenizer
        tokenized_text = tokenizer(text)['input_ids']
        if len(tokenized_text) < tokenizer.model_max_length:
            return [tokenized_text]
        else:
            first_chunk, second_chunk = split_text_by_lines(text)
            return self.chunkify(first_chunk,level) + self.chunkify(second_chunk,level)

def split_text_by_lines(text):
    lines = text.split('\n')
    N = len(text.split())
    first_chunk_size = 0
    for i,l in enumerate(lines):
        first_chunk_size += len(l.split())
        if first_chunk_size > N/2:
            break
    first_chunk = '\n'.join(lines[:i+1])
    second_chunk = '\n'.join(lines[i+1:])
    return first_chunk, second_chunk

def get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return {'r'+k[5:]+a: round(getattr(v,a),4) for k,v in raw_rscores.items() for a in ('precision','recall','fmeasure')}

def harmonic_avg(args):
    if any([a==0 for a in args]):
        print(args)
        return 0
    return len(args)/sum([1/x for x in args])

if __name__ == '__main__':
    import openai
    import argparse

    openai.api_key = "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"
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dpoints','-n',type=int,default=2)
    parser.add_argument('--do_shuffle',action='store_true')
    parser.add_argument('--do_check_gpt',action='store_true')
    parser.add_argument('--only_check_gpt',action='store_true')
    parser.add_argument('--summ_scenes_only',action='store_true')
    ARGS = parser.parse_args()

    if ARGS.only_check_gpt:
        ARGS.do_check_gpt = True

    all_our_bests = {}
    all_gpt_bests = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not ARGS.only_check_gpt:
        ss = SoapSummer(device)
    all_ep_names = os.listdir('SummScreen/transcripts')
    assert all([x.endswith('.json') for x in all_ep_names])
    all_ep_names = [x[:-5] for x in all_ep_names]
    all_ep_names.remove('oltl-10-18-10')
    all_ep_names.insert(0,'oltl-10-18-10')
    if ARGS.do_shuffle:
        np.random.shuffle(all_ep_names)
    n_procced = 0
    for ep_name in all_ep_names:
        if n_procced == ARGS.n_dpoints:
            break
        print(n_procced,ep_name)
        with open(join('SummScreen/transcripts',f'{ep_name}.json')) as f:
            transcript_data = json.load(f)
        with open(join('SummScreen/summaries',f'{ep_name}.json')) as f:
            summary_data = json.load(f)

        ep = Episode(ep_name,transcript_data,summary_data)

        if not '[SCENE_BREAK]' in transcript_data['Transcript']:
            continue
        else:
            n_procced += 1
        ep = episode_from_ep_name(ep_name)
        if ARGS.summ_scenes_only:
            ss.get_scene_summs(ep_name)
            continue
        else:
            n_procced += 1
        if not ARGS.only_check_gpt:
            summ_of_summs = ss.summarize(ep)
        if ARGS.do_check_gpt:
            gpt_summ = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Please summarize the following TV show {ep.transcript}"},])['choices'][0]['message']['content']
        our_best = -1
        gpt_best = -1
        for summ_name, gt_summ in ep.summaries.items():
            print('\n'+summ_name)
            print('Summary of summaries:')
            if not ARGS.only_check_gpt:
                our_scores = get_rouges(summ_of_summs,gt_summ)
                our_avg = harmonic_avg([v for k,v in our_scores.items() if 'fmeasure' in k])
                if our_scores['r2fmeasure'] > our_best:
                    our_best_scores = our_scores
                    our_best = our_scores['r2fmeasure']
                print(our_scores)
            if ARGS.do_check_gpt:
                print('GPT:')
                gpt_scores = get_rouges(gpt_summ,gt_summ)
                gpt_avg = harmonic_avg([v for k,v in gpt_scores.items() if 'fmeasure' in k])
                if gpt_scores['r2fmeasure'] > gpt_best:
                    gpt_best_scores = gpt_scores
                    gpt_best = gpt_scores['r2fmeasure']
                print(gpt_scores)

    if (not ARGS.summ_scenes_only) and (not ARGS.only_check_gpt):
        our_df = pd.DataFrame(all_our_bests).T
        our_df.loc['mean']=our_df.mean(axis=0)
        our_df.to_csv('our_rouge_scores.csv')
        print(our_df.loc['mean'])
    if (not ARGS.summ_scenes_only) and ARGS.do_check_gpt:
        gpt_df = pd.DataFrame(all_gpt_bests).T
        gpt_df.loc['mean']=gpt_df.mean(axis=0)
        print(gpt_df.loc['mean'])
        gpt_df.to_csv('gpt_rouge_scores.csv')
