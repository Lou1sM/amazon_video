from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from dl_utils.misc import check_dir
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from reorder import optimal_order, names_in_scene, identical_char_names
from time import time
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode, episode_from_epname
from utils import chunkify, summ_short_scene, safe_decode, rouge_from_multiple_refs
import numpy as np
from random import shuffle
from tqdm import tqdm
from dl_utils.torch_misc import show_gpu_memory


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SoapSummer():
    def __init__(self, model, tokenizer, caps, do_reorder, expname, do_resumm_scenes=False, do_save_new_scenes=False, is_test=False):
        self.model = model
        assert isinstance(expname,str)
        self.expname = expname
        self.n_epochs = 0
        self.dtokenizer = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
        self.tokenizer = tokenizer
        self.dmodel = AutoModelForSeq2SeqLM.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary').to(device)
        self.caps = caps
        self.do_reorder = do_reorder
        self.do_resumm_scenes = do_resumm_scenes
        self.do_save_new_scenes = do_save_new_scenes
        self.is_test = is_test
        self.bs = 1
        self.dbs = 8

    def pad_batch(self,batch,tokenizer):
        N=max([len(c) for c in batch])
        attention_mask = torch.stack([torch.cat([torch.ones(len(c)),torch.zeros(N-len(c))]) for c in batch], dim=0).cuda()
        padded = [b+[tokenizer.eos_token_id]*(N-len(b)) for b in batch]
        padded = torch.tensor(padded).to(device)
        assert padded.shape == attention_mask.shape
        return padded, attention_mask

    def epname_in_context(self,epname):
        fn = epname + '_reordered' if self.do_reorder else epname
        if self.caps != 'nocaptions':
            fn += f'_{self.caps}caps'
        if self.is_test:
            fn += '_test'
        return fn

    def summ_scenes(self, ep):
        start_time = time()
        if len(ep.scenes) == 1:
            print(f'no scene breaks for {ep.ep_name}')
            breakpoint()
        if self.caps == 'nocaptions':
            caps = ['']*len(ep.scenes)
        else: # prepend vid caps to the scene summ
            with open(f'SummScreen/video_scenes/{ep.ep_name}/{self.caps}_procced_scene_caps.json') as f:
                caps_data = json.load(f)
            cdd = {c['scene_id']:c['with_names'] for c in caps_data}
            caps = [cdd.get(f'{ep.ep_name}s{i}','') for i in range(len(ep.scenes))]
            if not len(caps)==len(ep.scenes):
                breakpoint()
        if not all('talking' not in x for x in caps):
            breakpoint()
        if self.do_reorder:
            order_idxs = optimal_order(ep.scenes)
            optimally_ordered_scenes = [ep.scenes[oi] for oi in order_idxs[:-1]]
            optimally_ordered_caps = [caps[oi] for oi in order_idxs[:-1]]
            combined_scenes = [optimally_ordered_scenes[0]]
            combined_caps = [optimally_ordered_caps[0]]
            for optscene, optcap in zip(optimally_ordered_scenes[1:],optimally_ordered_caps[1:]):
                if identical_char_names(optscene, combined_scenes[-1]):
                    combined_scenes[-1]+=optscene.lstrip()
                    combined_caps[-1]+=optcap.lstrip()
                else:
                    combined_scenes.append(optscene)
                    combined_caps.append(optcap)
        else:
            combined_scenes = ep.scenes
            combined_caps = caps
        #print([names_in_scene(s) for s in combined_scenes])

        chunk_list = [chunkify(s,self.dtokenizer.model_max_length) for s in combined_scenes]
        chunks = sum(chunk_list,[])
        avg_scene_summ_len = self.tokenizer.model_max_length//len(chunks)

        tok_chunks = [self.dtokenizer(c)['input_ids'] for c in chunks]
        sort_idxs = np.argsort([len(x) for x in tok_chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_chunks = [chunks[i] for i in sort_idxs]
        sorted_tok_chunks = [tok_chunks[i] for i in sort_idxs]
        v_short_chunk_idxs = [i for i,sc in enumerate(sorted_tok_chunks) if len(sc) < avg_scene_summ_len]
        n_shorts = len(v_short_chunk_idxs)
        if n_shorts>0:
            print(f'averge scene summ length is {avg_scene_summ_len}, and shortest scene is of length {len(sorted_tok_chunks[0])} tokens, {n_shorts} will be short summed')
        assert v_short_chunk_idxs == list(range(n_shorts))
        #v_short_chunks = [sc for sc in sorted_chunks if len(sc) < avg_scene_summ_len]
        short_chunk_summs = [summ_short_scene(sc) for sc in sorted_chunks[:n_shorts]]
        remaining_chunks = sorted_tok_chunks[n_shorts:]
        assert all([sorted_tok_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(tok_chunks)])
        #N = ceil(len(chunks)/self.dbs)
        N = ceil(len(remaining_chunks)/self.dbs)
        remaining_chunk_summs = []
        for i in range(N):
            padded, attn = self.pad_batch(remaining_chunks[i*self.dbs:(i+1)*self.dbs],self.dtokenizer)
            max_len = min(padded.shape[1],avg_scene_summ_len+15)
            min_len = max(10,max_len-20)
            if padded.shape[1] > self.dtokenizer.model_max_length:
                print('too long', padded.shape, self.dtokenizer.model_max_length)
                padded = padded[:,:self.dtokenizer.model_max_length]
                attn = attn[:,:self.dtokenizer.model_max_length]
            #show_gpu_memory()
            summ_tokens = self.dmodel.generate(padded, attention_mask=attn, min_length=min_len, max_length=max_len)
            assert summ_tokens.shape[1] <= max_len
            summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            len_first_unpadded = attn[0].argmin()
            if len_first_unpadded==0:
                assert attn[0].all()
                len_first_unpadded = attn.shape[1]
            #if not summ[0] == self.dtokenizer.decode(self.dmodel.generate(padded[:1,:len_first_unpadded], min_length=min_len,max_length=max_len)[0], skip_special_tokens=True, clean_up_tokenization_spaces=True):
                #print(self.dmodel.generate(padded[:1,:len_first_unpadded], min_length=min_len,max_length=max_len)[0])
                #print(padded[0])
                #print(padded[0,:len_first_unpadded])
            remaining_chunk_summs += summ
        chunk_summs = short_chunk_summs + remaining_chunk_summs
        #print(f'Scene summ time: {time()-start_time:.2f}')

        # now reuinfy whatever scenes were split into chunks
        desorted_chunk_summs = [chunk_summs[i] for i in reversed_sort_idxs]
        count = 0
        desplit = []
        for cl in chunk_list: # take lens from original list, before sorting
            desplit.append(' '.join(desorted_chunk_summs[count:count+len(cl)]))
            count+=len(cl)
        assert (desplit==desorted_chunk_summs) or (set([len(x) for x in chunk_list])!=set([1]))
        # if some were chunked together, may differ because of the join
        ss_with_caps = [f'{sc} {x}' for sc,x in zip(combined_caps,desplit)]
        #if ep.ep_name == 'oltl-10-18-10' and len(''.join(ss_with_caps))!=3109:
        if self.caps == 'nocaptions':
            assert self.tokenizer.model_max_length + 15*len(chunks) >= len(self.dtokenizer(''.join(ss_with_caps))[0])
        return ss_with_caps

    def get_scene_summs(self, ep):
        fn = self.epname_in_context(ep.ep_name)
        #if ep.ep_name == 'oltl-10-18-10':
            #breakpoint()
        maybe_scene_summ_path = f'SummScreen/scene_summs/{fn}.txt'
        if os.path.exists(maybe_scene_summ_path) and not self.do_resumm_scenes:
            with open(maybe_scene_summ_path) as f:
                ss = f.readlines()
        else:
            ss = self.summ_scenes(ep)
            if self.do_save_new_scenes:
                with open(fpath:=f'SummScreen/scene_summs/{fn}.txt','w') as f:
                    f.write('\n'.join(ss))
                print('saving to',fpath)
        return ss

    def summarize_from_ep(self, ep):
        scene_summs = self.get_scene_summs(ep)
        return self.summarize_scene_summs('\n'.join(scene_summs))

    def summarize_scene_summs(self, concatted_scene_summs):
        chunks = chunkify(concatted_scene_summs,self.tokenizer.model_max_length)
        tok_chunks = [self.tokenizer(c)['input_ids'] for c in chunks]
        #max_len = 300
        #min_len = max(10,max_len-20)
        #min_len = 280
        pbatch, attn = self.pad_batch(tok_chunks,self.tokenizer)
        if pbatch.shape[1] > self.tokenizer.model_max_length:
            breakpoint()
            pbatch = pbatch[:,:self.tokenizer.model_max_length]
        meta_chunk_summs = self.model.generate(pbatch, attention_mask=attn, min_length=280, max_length=300)
        final_summ = ' '.join(self.tokenizer.batch_decode(meta_chunk_summs,skip_special_tokens=True))
        return concatted_scene_summs, final_summ

    def dpoints_from_ep_names(self, ep_name_list, scene_caps):
        assert not any(['reordered' in x for x in ep_name_list])
        data_list = []
        #summer = SoapSummer(None, None, caps=scene_caps, do_reorder=self.do_reorder, do_resumm_scenes=do_resumm_scenes, do_save_new_scenes=not is_test)
        summ_dir = 'SummScreen/summaries'
        for ep_name in tqdm(ep_name_list):
            #show_gpu_memory()
            ep = episode_from_epname(ep_name)
            ss = ''.join(self.get_scene_summs(ep))
            with open(os.path.join(summ_dir, f'{ep_name}.json')) as f:
                d = json.load(f)
            if len(d.items())==0:
                breakpoint()
            for k,v in d.items():
                if '[ RECAP AVAILABLE ]' in v or 'Episode summary coming soon.' in v:
                    continue
                assert (k=='tvmega_summary') == (v.startswith('Episode'))
                if len(v) > 0 and k not in ['soap_central','tvmega_summary']:
                    data_list.append({'scene_summs':ss, 'summ':v, 'summ_name':k, 'ep_name':ep_name})
        return data_list

    def build_dset(self, scene_caps, n_dpoints):
        assert type(n_dpoints)==int
        fn = f'{scene_caps}_reordered' if self.do_reorder else scene_caps
        if self.is_test:
            fn += '_test'
        ep_names = [x.split('.')[0] for x in os.listdir('SummScreen/summaries') if os.path.getsize(f'SummScreen/summaries/{x}') > 100]
        print(len(ep_names),len(os.listdir('SummScreen/summaries')))
        if scene_caps != 'nocaptions':
            ep_names = [x for x in ep_names if os.path.exists(f'SummScreen/video_scenes/{x}/{scene_caps}_procced_scene_caps.json')]
        #ep_names = [x.replace('_reordered','') for x in os.listdir(scene_summ_dir)]
        #assert all([x.endswith('.txt') for x in ep_names])
        assert all([os.path.isdir(f'SummScreen/video_scenes/{x}') for x in ep_names])
        #ep_names = [x[:-4] for x in ep_names]
        shuffle(ep_names)
        ep_name_to_be_first = 'atwt-01-02-03'
        ep_names.remove(ep_name_to_be_first)
        if n_dpoints != -1:
            ep_names = ep_names[:n_dpoints]
        tr_up_to_idx = int(9*len(ep_names)/10)
        tr_ep_names = ep_names[:tr_up_to_idx]
        ts_ep_names = ep_names[tr_up_to_idx:]
        if self.is_test:
            tr_ep_names = tr_ep_names[:20]
            ts_ep_names = ts_ep_names[:3]
        ts_ep_names.insert(0,ep_name_to_be_first)
        print('getting scene summs for train set')
        print('getting scene summs for test set')
        ts_list = self.dpoints_from_ep_names(ts_ep_names, scene_caps)
        #test_ep_names = set(x['ep_name'] for x in ts_list)
        #test_ep_names = ['atwt-05-11-05']+list(test_ep_names)
        #assert set(x['ep_name'] for x in ts_list).issubset(set(ts_ep_names)) #some have no summary
        if not set(x['ep_name'] for x in ts_list) == set(ts_ep_names):
            breakpoint()
        ts_combined_list = []
        for tepn in ts_ep_names:
            dps_with_name = [t for t in ts_list if t['ep_name']==tepn]
            assert all(d['scene_summs']==dps_with_name[0]['scene_summs'] for d in dps_with_name[1:])
            combined = {'ep_name':tepn, 'scene_summs': dps_with_name[0]['scene_summs']}
            for dpsn in dps_with_name:
                combined[dpsn['summ_name']] = dpsn['summ']
            if tepn == ep_name_to_be_first:
                to_be_first = combined
            else:
                ts_combined_list.append(combined)
        ts_combined_list.insert(0, to_be_first)
        check_dir('SummScreen/json_datasets')
        with open(f'SummScreen/json_datasets/test_{fn}_dset.json','w') as f:
            json.dump(ts_combined_list, f)

        tr_list = self.dpoints_from_ep_names(tr_ep_names, scene_caps)
        with open(f'SummScreen/json_datasets/train_{fn}_dset.json','w') as f:
            json.dump(tr_list, f)

    def train_one_epoch(self, epoch, trainloader):
        self.model.train()
        self.opt.zero_grad()
        epoch_loss = 0
        for i,batch in enumerate(pbar := tqdm(trainloader, dynamic_ncols=True, smoothing=0.01, leave=False)):
            if (l := batch['input_ids'].shape[1]) > self.tokenizer.model_max_length*6/5:
                #print('skipping because inputs are of length', l)
                continue
            else:
                batch['input_ids'] = batch['input_ids'][:,:self.tokenizer.model_max_length]
                batch['attention_mask'] = batch['attention_mask'][:,:self.tokenizer.model_max_length]
            if (l := batch['labels'].shape[1]) > self.tokenizer.model_max_length:
                print('skipping because labels are of length', l)
                continue
            #cbatch = {k:v.cuda()[:,:self.tokenizer.model_max_length] for k,v in batch.items()}
            if (x:=batch['input_ids'].shape[1]) + (y:=batch['labels'].shape[1]) > 1550:
                #print(f'skipping because inputs are {x} and labels are {y} so maybe give OOM')
                continue
            if max(batch['input_ids'].shape[1], batch['labels'].shape[1], batch['decoder_input_ids'].shape[1]) > self.tokenizer.model_max_length:
                breakpoint()
            cbatch = {k:v.cuda() for k,v in batch.items()}
            cbatch['labels'] = cbatch['labels'].contiguous()
            #print(safe_decode(cbatch['input_ids'], self.tokenizer),'\n',safe_decode(cbatch['labels'], self.tokenizer))
            #show_gpu_memory()
            #print(f'inputs: {len(cbatch["input_ids"][0])}\tlabels: {len(cbatch["labels"][0])}')
            try:
                outputs = self.model(**cbatch)
                #show_gpu_memory()
                loss = outputs[0]
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                print(f'got OOM with inputs {x} and labels {y}')
                continue

            epoch_loss = ((i*epoch_loss) + loss.item()) / (i+1) # running avg
            pbar.set_description(f'Epoch: {epoch}/{self.n_epochs}'
                                 f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
            self.opt.step(); self.lr_scheduler.step()
            #if i==ARGS.n_iter or (i==10 and ARGS.is_test):
            if i==10 and self.is_test:
                break
        return epoch_loss

    def save_to(self, fname):
        save_dir = join('experiments', self.expname, 'checkpoints', fname)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def eval_epoch(self, epoch_num, testset):
        self.model.eval()
        print('validating')
        prev = ''
        rouges = []
        epoch_rouge = np.zeros(3)
        check_dir(generations_dir := join('experiments', self.expname, 'generations'))
        for j,batch in enumerate(val_pbar := tqdm(testset, dynamic_ncols=True, smoothing=0.01, leave=False)):
            #tensor_inputs = torch.tensor(batch['input_ids'][:tokenizer.model_max_length],device=device)
            nl_inputs = batch['scene_summs']
            #outputs = model.generate(tensor_inputs.unsqueeze(0),min_length=250,max_length=300)
            _, nl_outputs = self.summarize_scene_summs(nl_inputs)
            if (nl_outputs[:100] == prev[:100]):# and not (prev_inp[:100] == batch['input_ids'][:100]):
                print('repeat output')
            #prev_inp = batch['input_ids']
            prev = nl_outputs
            references = [v for k,v in batch.items() if k not in ('ep_name','scene_summs') and v is not None]
            best_rouge = rouge_from_multiple_refs(nl_outputs, references, return_full=False, benchmark_rl=True)

            rouges.append(best_rouge)
            epoch_rouge = ((j*epoch_rouge) + best_rouge) / (j+1) # running avg
            val_pbar.set_description(f'Epoch: {epoch_num}/{self.n_epochs}'
                             f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f}  '
                             f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f}')
            ep_name = batch['ep_name']
            with open(f'{generations_dir}/{ep_name}.txt','w') as f:
                f.write(nl_outputs)
            if j==2 and self.is_test:
                break
        return rouges

    def train_epochs(self, n_epochs, trainset, testset, save_every, eval_every):
        self.opt = AdamW(self.model.model.decoder.parameters(),lr=1e-6)
        #self.expdir = join('experiments',expname)
        dc = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainloader = DataLoader(trainset, batch_size=self.bs, shuffle=True, collate_fn=dc)
        #self.testset = testset

        self.n_epochs = n_epochs
        num_training_steps = self.n_epochs * len(trainloader)
        self.lr_scheduler = get_scheduler(
    name="linear", optimizer=self.opt, num_warmup_steps=0, num_training_steps=num_training_steps
            )
        patience = 0
        alltime_best_rouges = np.zeros(3)
        for epoch in range(self.n_epochs):
            print(f'training epoch {epoch}')
            epoch_loss = self.train_one_epoch(epoch, trainloader)
            if save_every:
                self.save_to(f'epoch{epoch}')
            print(f'Epoch: {epoch}\tLoss: {epoch_loss:.5f}')
            if (epoch+1) % eval_every == 0:
                rouges = self.eval_epoch(epoch, testset)
                rouges_arr = np.array(rouges).mean(axis=0)
                if rouges_arr[2] > alltime_best_rouges[2]:
                    patience = 0
                    alltime_best_rouges = rouges_arr
                    self.save_to('best')
                else:
                    patience += 1
                print(f'Mean Rouge: {rouges_arr}\tPatience: {patience}')
                if patience == 2:
                    break
        return alltime_best_rouges


if __name__ == '__main__':
    import openai
    import argparse

    openai.api_key = "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"
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dpoints','-n', type=int, default=2)
    parser.add_argument('--do_shuffle', action='store_true')
    parser.add_argument('--do_check_gpt', action='store_true')
    parser.add_argument('--only_check_gpt', action='store_true')
    parser.add_argument('--summ_scenes_only', action='store_true')
    parser.add_argument('--do_resumm_scenes', action='store_true')
    parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions'],default='nocaptions')
    parser.add_argument('--do_reorder', action='store_true')
    parser.add_argument('-t','--is_test', action='store_true')
    ARGS = parser.parse_args()

    if ARGS.only_check_gpt:
        ARGS.do_check_gpt = True

    all_our_bests = {}
    all_csss_bests = {}
    all_gpt_bests = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not ARGS.only_check_gpt:

        model_name = 'lucadiliello/bart-small' if ARGS.is_test else 'facebook/bart-large-cnn'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        #    dtokenizer = tokenizer
        #    dmodel = model
        #else:
        #    model_name = 'facebook/bart-large-cnn'
        #    tokenizer = AutoTokenizer.from_pretrained(model_name)
        #    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        #    #dtokenizer = AutoTokenizer.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary')
        #    #dmodel = AutoModelForSeq2SeqLM.from_pretrained('kabita-choudhary/finetuned-bart-for-conversation-summary').to(device)

        ss = SoapSummer(model, tokenizer, ARGS.caps, ARGS.do_reorder, ARGS.do_resumm_scenes, ARGS.is_test)
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
        ep = episode_from_epname(ep_name)
        if ARGS.summ_scenes_only:
            ss.get_scene_summs(ep_name)
            continue
        if not ARGS.only_check_gpt:
            csss, summ_of_summs = ss.summarize_from_ep(ep)
        if ARGS.do_check_gpt:
            gpt_summ = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Please summarize the following TV show {ep.transcript}"},])['choices'][0]['message']['content']
        our_scores = ep.calc_rouge(summ_of_summs)
        all_our_bests[ep_name] = [our_scores[f'rouge-{x}']['f'] for x in (1,2,'l')]
        print(our_scores)
        #print('summ of summs:', our_scores)
        our_csss_scores = ep.calc_rouge(csss)
        all_csss_bests[ep_name] = [our_csss_scores[f'rouge-{x}']['f'] for x in (1,2,'l')]
        print(our_csss_scores)
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
