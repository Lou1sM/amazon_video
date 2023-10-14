from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reorder import optimal_order, names_in_scene, identical_char_names
from time import time
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode, episode_from_ep_name
#from torchmetrics.text.rouge import ROUGEScore
#from rouge_score import rouge_scorer
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

    def pad_batch(self,batch,eos_token_id):
        N=max([len(c) for c in batch])
        padded = [b+[eos_token_id]*(N-len(b)) for b in batch]
        return torch.tensor(padded).to(self.device)

    def summ_scenes(self,ep):
        start_time = time()
        if ARGS.tsp:
            order_idxs = optimal_order(ep.scenes)
            optimally_ordered_scenes = [ep.scenes[oi] for oi in order_idxs[:-1]]
            combined_scenes = [optimally_ordered_scenes[0]]
            for optscene in optimally_ordered_scenes[1:]:
                if identical_char_names(optscene, combined_scenes[-1]):
                    combined_scenes[-1]+=optscene.lstrip()
                else:
                    combined_scenes.append(optscene)
        else:
            combined_scenes = ep.scenes
        print([names_in_scene(s) for s in combined_scenes])
        chunk_list = [chunkify(s,self.dtokenizer.model_max_length) for s in combined_scenes]
        chunks = sum(chunk_list,[])
        tok_chunks = [self.dtokenizer(c)['input_ids'] for c in chunks]
        sort_idxs = np.argsort([len(x) for x in tok_chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_chunks = [tok_chunks[i] for i in sort_idxs]
        if not all([sorted_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(tok_chunks)]):
            breakpoint()
        N = ceil(len(chunks)/self.dbs)
        chunk_summs = []
        for i in range(N):
            padded = self.pad_batch(sorted_chunks[i*self.dbs:(i+1)*self.dbs],self.dtokenizer.eos_token_id)
            print(padded.shape)
            max_len = min(padded.shape[1],50)
            min_len = max(10,max_len-20)
            if padded.shape[1] > self.dtokenizer.model_max_length:
                print('too long', padded.shape, self.dtokenizer.model_max_length)
                padded = padded[:,:self.dtokenizer.model_max_length]
            summ_tokens = self.dmodel.generate(padded,min_length=min_len,max_length=max_len)
            summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=True)
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
        if os.path.exists(maybe_scene_summ_path) and not ARGS.resumm_scenes:
            with open(maybe_scene_summ_path) as f:
                return f.readlines()
        else:
            return self.summ_scenes(ep)

    def summarize(self,ep,recompute=False):
        start_time = time()
        scene_summs = self.get_scene_summs(ep.ep_name)
        concatted_scene_summs = '\n'.join(scene_summs)
        chunks = chunkify(concatted_scene_summs,self.tokenizer.model_max_length)
        tok_chunks = [self.tokenizer(c)['input_ids'] for c in chunks]
        max_len = 300
        min_len = max(10,max_len-20)
        meta_chunk_summs = self.model.generate(self.pad_batch(tok_chunks,self.tokenizer.eos_token_id),min_length=min_len,max_length=max_len)
        final_summ = ' '.join(self.tokenizer.batch_decode(meta_chunk_summs,skip_special_tokens=True))
        with open(f'SummScreen/full_summs/{ep.ep_name}.txt','w') as f:
            f.write(final_summ)
        print(f'Total summ time: {time()-start_time:.2f}')
        return concatted_scene_summs, final_summ

def chunkify(text,max_chunk_size):
    if len(text.split())*4/3 < max_chunk_size:
        return [text]
    else:
        first_chunk, second_chunk = split_text_by_sth(text)
        return chunkify(first_chunk,max_chunk_size) + chunkify(second_chunk,max_chunk_size)

def split_text_by_sth(text):
    for sep in ('\n', '. ', ', ', ' '):
        if sep in text.strip():
            return split_text_by_sep(text.strip(),sep)
    return text[:len(text)//2], text[len(text)//2:]

def split_text_by_sep(text,sep):
    lines = text.split(sep)
    N = len(text.split())
    first_chunk = ''
    for i,l in enumerate(lines):
        if abs(len((first_chunk+l).split()) - N/2) > abs(len(first_chunk.split())-N/2):
            break # get as close to halfway as possible
        if first_chunk=='':
            first_chunk = l
        else:
            first_chunk += sep+l
        assert text.startswith(first_chunk)
    second_chunk = text[len(first_chunk):]
    assert first_chunk+second_chunk == text
    return first_chunk, second_chunk

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
    parser.add_argument('--resumm_scenes',action='store_true')
    parser.add_argument('--tsp',action='store_true')
    ARGS = parser.parse_args()

    if ARGS.only_check_gpt:
        ARGS.do_check_gpt = True

    all_our_bests = {}
    all_csss_bests = {}
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
        print(n_procced)
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
        ep = episode_from_ep_name(ep_name)
        if ARGS.summ_scenes_only:
            ss.get_scene_summs(ep_name)
            continue
        if not ARGS.only_check_gpt:
            csss, summ_of_summs = ss.summarize(ep)
            print(summ_of_summs)
        if ARGS.do_check_gpt:
            gpt_summ = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Please summarize the following TV show {ep.transcript}"},])['choices'][0]['message']['content']
        our_scores = ep.calc_rouge(summ_of_summs)
        all_our_bests[ep_name] = our_scores
        #print('summ of summs:', our_scores)
        our_csss_scores = ep.calc_rouge(csss)
        all_csss_bests[ep_name] = our_csss_scores
        #print('concat of summs:', our_csss_scores)
        if ARGS.do_check_gpt:
            gpt_scores = ep.calc_rouge(gpt_summ)
            print('GPT:', gpt_scores)
            all_gpt_bests[ep_name] = our_scores
        n_procced += 1

    csss_df = pd.DataFrame(all_csss_bests).T
    csss_df.loc['mean']=csss_df.mean(axis=0)
    print(csss_df.loc['mean'])
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
