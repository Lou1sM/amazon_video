from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from time import time
import re
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode
#from torchmetrics.text.rouge import ROUGEScore
from rouge_score import rouge_scorer
import numpy as np


class SoapSummer():
    def __init__(self,device):
        self.device = device
        self.dtokenizer = AutoTokenizer.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary")
        self.dmodel = AutoModelForSeq2SeqLM.from_pretrained("kabita-choudhary/finetuned-bart-for-conversation-summary").to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(self.device)

        self.bs = 8
        self.dbs = 8

    def pad_batch(self,batch):
        N=max([len(c) for c in batch])
        padded = [b+[self.dtokenizer.eos_token_id]*(N-len(b)) for b in batch]
        return torch.tensor(padded).to(self.device)

    def summarize(self,ep):
        start_time = time()
        chunks = sum([self.chunkify(s,level='dialogue') for s in ep.scenes],[])
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
        concatted_scene_summs = '\n'.join(desorted_chunk_summs)
        chunks = self.chunkify(concatted_scene_summs,level='meta')
        assert len(chunks) < self.bs
        mean_gt_summ_len = sum([len(s.split()) for s in ep.summaries.values()])/len(ep.summaries) # soap_central v long, kinda skewing it
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

def summarize_scenes(scene_texts):
    text = text.replace('@@ ','').replace(' ,',',')
    max_len = min(len(text.split()),50)
    min_len = max(10,max_len-20)
    print(len(text.split()))
    if len(text.split())>800:
        first_chunk, second_chunk = split_text_by_lines(text)
        return summarize_scene(first_chunk) + summarize_scene(second_chunk)
    try:
        return dpipe(text,min_length=min_len, max_length=max_len)[0]['summary_text']
    except IndexError:
        first_chunk, second_chunk = split_text_by_lines(text)
        return summarize_scene(first_chunk) + summarize_scene(second_chunk)

def get_rouges(pred,gt):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL','rougeLsum'], use_stemmer=True)
    raw_rscores = scorer.score(gt,pred)
    return {'r'+k[5:]+a: round(getattr(v,a),4) for k,v in raw_rscores.items() for a in ('precision','recall','fmeasure')}

def get_summ_of_summs(concatted_summs,gt_len):
    assert type(gt_len) in (int,float)
    if len(concatted_summs.split())>800:
        first_chunk, second_chunk = split_text_by_lines(concatted_summs)
        return get_summ_of_summs(first_chunk,gt_len/2) + get_summ_of_summs(second_chunk,gt_len/2)
    max_len = int(min(gt_len-50,250))
    min_len = int(max(90,max_len-40))
    summ_of_summs = pipe(concatted_summs,min_length=min_len, max_length=max_len)[0]['summary_text']
    return summ_of_summs

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
    ARGS = parser.parse_args()

    if ARGS.only_check_gpt:
        ARGS.do_check_gpt = True

    all_our_bests = {}
    all_gpt_bests = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not ARGS.only_check_gpt:
        ss = SoapSummer(device)
    all_ep_fnames = os.listdir('SummScreen/transcripts')
    if ARGS.do_shuffle:
        np.random.shuffle(all_ep_fnames)
    for ep_fname in all_ep_fnames:

        with open(join('SummScreen/transcripts',ep_fname)) as f:
            transcript_data = json.load(f)
        if not '[SCENE_BREAK]' in transcript_data['Transcript']: continue
        with open(join('SummScreen/summaries',ep_fname)) as f:
            summary_data = json.load(f)

        ep = Episode(ep_fname,transcript_data,summary_data)

        #concatted_scene_summs = '\n'.join([summarize_scene(x) for x in ep.scenes])
        print('Concatted scene summaries:')
        #gt_len = sum([len(x.split()) for x in ep.summaries.values()])/len(ep.summaries)
        #summ_of_summs = get_summ_of_summs(concatted_scene_summs,gt_len)
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

        if not ARGS.only_check_gpt:
            print(f'\nBest ours: {our_best_scores}')
            all_our_bests[ep.show_name] = our_best_scores
        if ARGS.do_check_gpt:
            print(f'Best GPT: {gpt_best_scores}')
            all_gpt_bests[ep.title] = gpt_best_scores
        if (len(all_our_bests)==ARGS.n_dpoints) or (len(all_gpt_bests)==ARGS.n_dpoints and ARGS.only_check_gpt): break
    if not ARGS.only_check_gpt:
        our_df = pd.DataFrame(all_our_bests).T
        our_df.loc['mean']=our_df.mean(axis=0)
        our_df.to_csv('our_rouge_scores.csv')
        print(our_df.loc['mean'])
    if ARGS.do_check_gpt:
        gpt_df = pd.DataFrame(all_gpt_bests).T
        gpt_df.loc['mean']=gpt_df.mean(axis=0)
        print(gpt_df.loc['mean'])
        gpt_df.to_csv('gpt_rouge_scores.csv')
