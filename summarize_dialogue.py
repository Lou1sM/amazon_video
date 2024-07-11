from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from sklearn.feature_extraction.text import TfidfVectorizer
from dl_utils.misc import check_dir
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from reorder import optimal_order, identical_char_names
import torch
from math import ceil
import pandas as pd
import os
from os.path import join
import json
from episode import Episode, episode_from_epname
from utils import chunkify, summ_short_scene, rouge_from_multiple_refs, get_fn
import numpy as np
from tqdm import tqdm


class SoapSummer():
    def __init__(self, model_name, device, bs, dbs, caps, scene_order, uniform_breaks, startendscenes, centralscenes, max_chunk_size, expdir, data_dir, resumm_scenes=False, do_save_new_scenes=False, is_test=False):
        assert not (centralscenes and startendscenes)
        assert isinstance(expdir,str)
        self.device = device
        if is_test:
            self.model_name = self.dmodel_name = 'lucadiliello/bart-small'
        elif startendscenes or centralscenes:
            self.model_name = self.dmodel_name = 'kabita-choudhary/finetuned-bart-for-conversation-summary'
        else:
            self.dmodel_name = 'kabita-choudhary/finetuned-bart-for-conversation-summary'
            self.model_name = model_name
        self.dmodel = AutoModelForSeq2SeqLM.from_pretrained(self.dmodel_name).to(device)
        self.dtokenizer = AutoTokenizer.from_pretrained(self.dmodel_name)
        if self.model_name == self.dmodel_name: # either test, or startend/central
            self.model = self.dmodel # avoid loading twice
            self.tokenizer = self.dtokenizer # avoid loading twice
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if caps.endswith('-only'):
            self.caps = caps[:-5]
            self.caps_only = True
        else:
            self.caps = caps
            self.caps_only = False
        self.expdir = expdir
        self.data_dir = data_dir
        self.n_epochs = 0
        self.scene_order = scene_order
        self.uniform_breaks = uniform_breaks
        self.startendscenes = startendscenes
        self.centralscenes = centralscenes
        self.max_chunk_size = max_chunk_size
        self.resumm_scenes = resumm_scenes
        self.do_save_new_scenes = do_save_new_scenes
        self.is_test = is_test
        self.bs = bs
        self.dbs = dbs
        self.fn = get_fn(caps, self.scene_order, self.uniform_breaks, self.startendscenes, self.centralscenes, self.is_test)

    def pad_batch(self,batch,tokenizer):
        N=max([len(c) for c in batch])
        attention_mask = torch.stack([torch.cat([torch.ones(len(c)),torch.zeros(N-len(c))]) for c in batch], dim=0).to(self.device)
        padded = [b+[tokenizer.eos_token_id]*(N-len(b)) for b in batch]
        padded = torch.tensor(padded).to(self.device)
        assert padded.shape == attention_mask.shape
        return padded, attention_mask

    def summ_scenes(self, epname, infer_splits):
        ep = episode_from_epname(epname, infer_splits)
        scenes = ['']*len(ep.scenes) if self.caps_only else ep.scenes
        max_chunk_size = min(self.max_chunk_size, self.dtokenizer.model_max_length)
        if len(scenes) == 1:
            print(f'no scene breaks for {epname}')
            breakpoint()
        if self.caps == 'nocaptions':
            caps = ['']*len(scenes)
        else: # prepend vid caps to the scene summ
            with open(join(self.data_dir,f'video_scenes/{epname}/{self.caps}_procced_scene_caps.json')) as f:
                caps_data = json.load(f)
            cdd = {c['scene_id']:c['with_names'] for c in caps_data}
            caps = [cdd.get(f'{epname}s{i}','') for i in range(len(scenes))]
            if not len(caps)==len(scenes):
                breakpoint()
        assert all('talking' not in x for x in caps)
        if self.scene_order=='optimal':
            order_idxs = optimal_order(scenes)
            optimally_ordered_scenes = [scenes[oi] for oi in order_idxs[:-1]]
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
        elif self.scene_order=='rand':
            idxs = sorted(range(len(scenes)), key=lambda x: np.random.rand())
            combined_scenes = [scenes[ri] for ri in idxs]
            combined_caps = [caps[ri] for ri in idxs]
        elif self.uniform_breaks:
            transcript_wo_scene_marks = '\n'.join([x for x in ep.transcript if x!='[SCENE_BREAK]'])
            combined_scenes = chunkify(transcript_wo_scene_marks, max_chunk_size)
            combined_caps = caps
        else:
            combined_scenes = scenes
            combined_caps = caps

        if self.startendscenes:
            start = ''
            end = ''
            newend = ''
            startidx = 0
            endidx = 0
            while True:
                if startidx == endidx:
                    newstart = start + scenes[startidx]
                    startidx += 1
                else:
                    assert startidx == endidx+1
                    newend = scenes[-endidx-1] + end
                    endidx += 1
                if len((newstart+newend).split()) > 3/4*self.tokenizer.model_max_length:
                    break
                else:
                    start, end = newstart, newend
            assert len((start+end).split())*4/3 <= self.tokenizer.model_max_length
            return start + end
        elif self.centralscenes:
            vect = TfidfVectorizer(min_df=1, stop_words='english')
            tfidf = vect.fit_transform(combined_scenes)
            scene_sims = (tfidf * tfidf.T).A
            np.fill_diagonal(scene_sims, -1)
            y = scene_sims
            best_scene_idxs = y.sum(axis=0).argsort()[::-1]
            for j in range(100):
                y = np.matmul(y,y.T)
                new_best_scene_idxs = y.sum(axis=0).argsort()[::-1]
                if (new_best_scene_idxs == best_scene_idxs).all():
                    break
                else:
                    best_scene_idxs = new_best_scene_idxs
            for go_up_to in range(len(best_scene_idxs)):
                if sum(len(scenes[sidx].split()) for sidx in best_scene_idxs[:go_up_to+1])*4/3 > self.dtokenizer.model_max_length:
                    break # next-one-up would push it over the edge
            idxs_to_use = sorted(best_scene_idxs[:go_up_to])

            best_in_order = [scenes[i] for i in idxs_to_use]
            assert sum(len(x.split()) for x in best_in_order)*4/3 <= self.dtokenizer.model_max_length
            return best_in_order

        if self.caps_only:
            return combined_caps
        chunk_list = [chunkify(s, max_chunk_size) for s in combined_scenes]
        chunks = sum(chunk_list,[])
        assert (chunks==combined_scenes) or not self.uniform_breaks
        avg_scene_summ_len = self.tokenizer.model_max_length//len(chunks)

        tok_chunks = [self.dtokenizer(c)['input_ids'] for c in chunks]
        sort_idxs = np.argsort([len(x) for x in tok_chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_chunks = [chunks[i] for i in sort_idxs]
        sorted_tok_chunks = [tok_chunks[i] for i in sort_idxs]
        v_short_chunk_idxs = [i for i,sc in enumerate(sorted_tok_chunks) if len(sc) < avg_scene_summ_len]
        n_shorts = len(v_short_chunk_idxs)
        assert v_short_chunk_idxs == list(range(n_shorts))
        short_chunk_summs = [summ_short_scene(sc) for sc in sorted_chunks[:n_shorts]]
        remaining_chunks = sorted_tok_chunks[n_shorts:]
        assert all([sorted_tok_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(tok_chunks)])
        N = ceil(len(remaining_chunks)/self.dbs)
        remaining_chunk_summs = []
        for i in range(N):
            padded, attn = self.pad_batch(remaining_chunks[i*self.dbs:(i+1)*self.dbs],self.dtokenizer)
            max_len = min(padded.shape[1],avg_scene_summ_len+20)
            min_len = max(10,max_len-20)
            if padded.shape[1] > self.dtokenizer.model_max_length:
                padded = padded[:,:self.dtokenizer.model_max_length]
                attn = attn[:,:self.dtokenizer.model_max_length]
            summ_tokens = self.dmodel.generate(padded, attention_mask=attn, min_length=min_len, max_length=max_len)
            assert summ_tokens.shape[1] <= max_len
            summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            len_first_unpadded = attn[0].argmin()
            if len_first_unpadded==0:
                assert attn[0].all()
                len_first_unpadded = attn.shape[1]
            remaining_chunk_summs += summ
        chunk_summs = short_chunk_summs + remaining_chunk_summs
        chunk_summs = [drop_trailing_halfsent(cs) for cs in chunk_summs]

        # return chunks to their original order
        desorted_chunk_summs = [chunk_summs[i] for i in reversed_sort_idxs]
        count = 0
        desplit = []
        # recombine scenes whose dialogue was split because of context size
        for cl in chunk_list: # take lens from original list, before it was sorted
            desplit.append(' '.join(desorted_chunk_summs[count:count+len(cl)]))
            count+=len(cl)
        assert (desplit==desorted_chunk_summs) or (set([len(x) for x in chunk_list])!=set([1]))
        # if some were chunked together, may differ because of the join
        ss_with_caps = [f'{sc} {x}' for sc,x in zip(combined_caps,desplit)]
        if self.caps == 'nocaptions':
            assert self.tokenizer.model_max_length + 15*len(chunks) >= len(self.dtokenizer(''.join(ss_with_caps))[0])
        return ss_with_caps

    def get_scene_summs(self, epname, infer_splits):
        epname_ = f'{epname}-inferred' if infer_splits else epname
        maybe_scene_summ_path = join(self.data_dir,f'scene_summs/{epname_}_{self.fn}.txt')
        if os.path.exists(maybe_scene_summ_path) and not self.resumm_scenes:
            with open(maybe_scene_summ_path) as f:
                ss = [x.strip() for x in f.readlines()]
        else:
            ss = self.summ_scenes(epname, infer_splits)
            if self.do_save_new_scenes:
                with open(maybe_scene_summ_path,'w') as f:
                    f.write('\n'.join(ss))
        return ss

    def summarize_from_epname(self, epname, min_len, max_len):
        scene_summs = self.get_scene_summs(epname, infer_splits=False)
        return self.summarize_scene_summs('\n'.join(scene_summs), min_len, max_len)

    def summarize_scene_summs(self, concatted_scene_summs, min_len=360, max_len=400):
        max_chunk_size = min(self.max_chunk_size, self.tokenizer.model_max_length)
        chunks = chunkify(concatted_scene_summs, max_chunk_size)
        tok_chunks = [self.tokenizer(c)['input_ids'] for c in chunks]
        pbatch, attn = self.pad_batch(tok_chunks,self.tokenizer)
        pbatch = pbatch[:,:self.tokenizer.model_max_length]
        attn = attn[:,:self.tokenizer.model_max_length]
        if self.caps_only:
            min_chunk_len = 80
            max_chunk_len = 100
        else:
            min_chunk_len = min_len//len(chunks) + 5 # +5 cuz will cut off incomplete sents
            max_chunk_len = max_len//len(chunks) + 5
        meta_chunk_summs = self.model.generate(pbatch, attention_mask=attn, min_length=min_chunk_len, max_length=max_chunk_len)
        text_mcss = self.tokenizer.batch_decode(meta_chunk_summs,skip_special_tokens=True)
        text_mcss = [drop_trailing_halfsent(tmcs) for tmcs in text_mcss]
        final_summ = ' '.join(text_mcss)
        return concatted_scene_summs, final_summ

    def dpoints_from_epnames(self, epname_list, scene_caps, infer_splits):
        assert not any(['reordered' in x for x in epname_list])
        data_list = []
        summ_dir = 'SummScreen/summaries'
        pbar = tqdm(epname_list)
        for epname in pbar:
            pbar.set_description(epname)
            unjoined_scene_summs = self.get_scene_summs(epname, infer_splits)
            ss = '\n'.join(unjoined_scene_summs)
            with open(os.path.join(summ_dir, f'{epname}.json')) as f:
                d = json.load(f)
            if len(d.items())==0:
                breakpoint()
            for k,v in d.items():
                if '[ RECAP AVAILABLE ]' in v or 'Episode summary coming soon.' in v:
                    continue
                assert (k=='tvmega_summary') == (v.startswith('Episode'))
                data_list.append({'scene_summs':ss, 'summ':v, 'summ_name':k, 'epname':epname})
        return data_list

    def build_dset(self, scene_caps, n_dpoints, dset_fragment_name):
        dset_info = pd.read_csv('dset_info.csv', index_col=0)
        base_dset_fragment_name = dset_fragment_name.removesuffix('-inferred')
        mask = dset_info['usable'] & (dset_info['split']==base_dset_fragment_name)
        epnames = dset_info.index[mask]
        epname_to_be_first = 'oltl-10-18-10'
        if n_dpoints != -1:
            if base_dset_fragment_name in ('val', 'test'):
                ndps_to_use = max(2,int(n_dpoints/10))
            else:
                assert dset_fragment_name == 'train'
                ndps_to_use = n_dpoints - 2*max(2,int(n_dpoints/10))
            assert ndps_to_use >= 2
            epnames = epnames[:ndps_to_use]
        if base_dset_fragment_name == 'test':
            epnames.insert(0, epname_to_be_first)
        else:
            epnames = [x for x in epnames if x!=epname_to_be_first]

        assert all([os.path.isdir(join(self.data_dir,f'video_scenes/{x}')) for x in epnames])
        infer_splits = dset_fragment_name.endswith('-inferred')
        dpoints = self.dpoints_from_epnames(epnames, scene_caps, infer_splits)
        if base_dset_fragment_name in ('val', 'test'):
            todump = []
            for tepn in epnames:
                dps_with_name = [t for t in dpoints if t['epname']==tepn]
                assert all(d['scene_summs']==dps_with_name[0]['scene_summs'] for d in dps_with_name[1:])
                combined = {'epname':tepn, 'scene_summs': dps_with_name[0]['scene_summs']}
                for dpsn in dps_with_name:
                    combined[dpsn['summ_name']] = dpsn['summ']
                todump.append(combined)
        else:
            todump = dpoints
        with open(join(self.data_dir,f'json_datasets/{self.fn}_{n_dpoints}dps_{dset_fragment_name}_dset.json','w')) as f:
            json.dump(todump, f)

    def train_one_epoch(self, epoch, trainloader, no_pbar):
        self.model.train()
        self.opt.zero_grad()
        epoch_loss = 0
        if no_pbar:
            trainiter = trainloader
        else:
            trainiter = tqdm(trainloader, dynamic_ncols=True, smoothing=0.01, leave=False)
        for i,batch in enumerate(trainiter):
            if (batch['input_ids'].shape[1]) > self.tokenizer.model_max_length*6/5 and not self.is_test:
                continue
            else:
                batch['input_ids'] = batch['input_ids'][:,:self.tokenizer.model_max_length]
                batch['attention_mask'] = batch['attention_mask'][:,:self.tokenizer.model_max_length]
                assert 'inputs_embeds' not in batch.keys()
            if (batch['labels'].shape[1]) > self.tokenizer.model_max_length:
                continue
            if max(batch['input_ids'].shape[1], batch['labels'].shape[1], batch['decoder_input_ids'].shape[1]) > self.tokenizer.model_max_length:
                breakpoint()
            cbatch = {k:v.to(self.device) for k,v in batch.items()}
            cbatch['labels'] = cbatch['labels'].contiguous()
            try:
                outputs = self.model(**cbatch)
                loss = outputs[0]
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                x=batch['input_ids'].shape[1]
                y=batch['labels'].shape[1]
                print(f'got OOM with inputs {x} and labels {y}')
                continue
            epoch_loss = ((i*epoch_loss) + loss.item()) / (i+1) # running avg
            if not no_pbar:
                trainiter.set_description(f'Epoch: {epoch}/{self.n_epochs}'
                                 f'current loss: {loss.item():.4f}  epoch loss: {epoch_loss:.4f}')
            self.opt.step(); self.lr_scheduler.step()
            if i==10 and self.is_test:
                break
        return epoch_loss

    def save_to(self, fname):
        save_dir = join(self.expdir, 'checkpoints', fname)
        print('saving checkpoint to',save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def inference_epoch(self, epoch_num, dset, dset_fragment_name):
        self.model.eval()
        print('validating')
        prev = ''
        rouges = []
        epoch_rouge = np.zeros(4)
        check_dir(generations_dir := join(self.expdir, f'generations_{dset_fragment_name}'))
        for j,batch in enumerate(pbar := tqdm(dset, dynamic_ncols=True, smoothing=0.01, leave=False)):
            nl_inputs = batch['scene_summs']
            _, nl_outputs = self.summarize_scene_summs(nl_inputs)
            if (nl_outputs[:100] == prev[:100]):# and not (prev_inp[:100] == batch['input_ids'][:100]):
                print('repeat output')
            prev = nl_outputs
            references = [v for k,v in batch.items() if k not in ('epname','scene_summs') and v is not None]
            best_rouge = rouge_from_multiple_refs(nl_outputs, references, return_full=False, benchmark_rl=True)

            rouges.append(best_rouge)
            epoch_rouge = ((j*epoch_rouge) + best_rouge) / (j+1) # running avg
            pbar.set_description(f'Epoch: {epoch_num}/{self.n_epochs}'
                             f'current rouge: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f} {best_rouge[3]:.3f}  '
                             f'epoch rouge: {epoch_rouge[0]:.3f} {epoch_rouge[1]:.3f} {epoch_rouge[2]:.3f} {epoch_rouge[3]:.3f}')
            epname = batch['epname']
            with open(f'{generations_dir}/{epname}.txt','w') as f:
                f.write(nl_outputs)
            if j==2 and self.is_test:
                break
        return np.array(rouges).mean(axis=0)

    def train_epochs(self, n_epochs, trainset, valset, testset, no_pbar, early_stop_metric):
        self.opt = AdamW(self.model.model.decoder.parameters(),lr=1e-6)
        dc = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        trainloader = DataLoader(trainset, batch_size=self.bs, shuffle=True, collate_fn=dc)
        self.n_epochs = n_epochs
        num_training_steps = self.n_epochs * len(trainloader)
        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.opt, num_warmup_steps=0, num_training_steps=num_training_steps)
        patience = 0
        alltime_best_rouges = np.zeros(4)
        all_rouges = []
        for epoch in range(self.n_epochs):
            print(f'training epoch {epoch}')
            epoch_loss = self.train_one_epoch(epoch, trainloader, no_pbar)
            print(f'Epoch: {epoch}\tLoss: {epoch_loss:.5f}')
            rouges_arr = self.inference_epoch(epoch, valset, 'val')
            #rouges = self.inference_epoch(epoch, valset, 'val')
            #rouges_arr = np.array(rouges).mean(axis=0)
            if len(all_rouges)>0 and (rouges_arr==all_rouges[-1]).all():
                print('WARNING: rouge unchanged since last epoch')
            else:
                assert not any((r==rouges_arr).all() for r in all_rouges)
            all_rouges.append(rouges_arr)
            if rouges_arr[early_stop_metric] > alltime_best_rouges[early_stop_metric]:
                patience = 0
                alltime_best_rouges = rouges_arr
                self.save_to('best')
            else:
                patience += 1
            print(f'Mean Rouge: {rouges_arr}\tPatience: {patience}')
            if patience == 2:
                break
        if self.n_epochs>0:
            best_chkpt = f'{self.expdir}/checkpoints/best'
            print('reloading', best_chkpt)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(best_chkpt).to(self.device)
        test_rouges = self.inference_epoch(self.n_epochs, testset, 'test')
        return test_rouges, alltime_best_rouges, all_rouges

def drop_trailing_halfsent(s):
    return '. '.join(x for x in s.split('. '))

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
    parser.add_argument('--resumm_scenes', action='store_true')
    parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions'],default='nocaptions')
    parser.add_argument('--order', type=str, choices=['identity','optimal','rand'], default='identity')
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

        ss = SoapSummer(model, tokenizer, ARGS.caps, ARGS.order, ARGS.resumm_scenes, ARGS.is_test)
    all_epnames = os.listdir('SummScreen/transcripts')
    assert all([x.endswith('.json') for x in all_epnames])
    all_epnames = [x[:-5] for x in all_epnames]
    all_epnames.remove('oltl-10-18-10')
    all_epnames.insert(0,'oltl-10-18-10')
    if ARGS.do_shuffle:
        np.random.shuffle(all_epnames)
    n_procced = 0
    for epname in all_epnames:
        if n_procced == ARGS.n_dpoints:
            break
        print(n_procced,epname)
        with open(join('SummScreen/transcripts',f'{epname}.json')) as f:
            transcript_data = json.load(f)
        with open(join('SummScreen/summaries',f'{epname}.json')) as f:
            summary_data = json.load(f)

        ep = Episode(epname,transcript_data,summary_data)

        if not '[SCENE_BREAK]' in transcript_data['Transcript']:
            continue
        ep = episode_from_epname(epname)
        if ARGS.summ_scenes_only:
            ss.get_scene_summs(epname)
            continue
        if not ARGS.only_check_gpt:
            csss, summ_of_summs = ss.summarize_from_epname(epname)
        if ARGS.do_check_gpt:
            gpt_summ = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f"Please summarize the following TV show {ep.transcript}"},])['choices'][0]['message']['content']
        our_scores = ep.calc_rouge(summ_of_summs)
        all_our_bests[epname] = [our_scores[f'rouge-{x}']['f'] for x in (1,2,'l')]
        print(our_scores)
        #print('summ of summs:', our_scores)
        our_csss_scores = ep.calc_rouge(csss)
        all_csss_bests[epname] = [our_csss_scores[f'rouge-{x}']['f'] for x in (1,2,'l')]
        print(our_csss_scores)
        #print('concat of summs:', our_csss_scores)
        if ARGS.do_check_gpt:
            gpt_scores = ep.calc_rouge(gpt_summ)
            print('GPT:', gpt_scores)
            all_gpt_bests[epname] = our_scores
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
