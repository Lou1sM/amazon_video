from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,  BitsAndBytesConfig
#from bitsandbytes import
from dl_utils.misc import check_dir
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from math import ceil
import os
from os.path import join
import json
from episode import episode_from_name
from utils import chunkify, rouge_from_multiple_refs, get_fn, get_all_testnames
import numpy as np
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftConfig


class SoapSummer():
    def __init__(self, device, bs, dbs, caps, scene_order, uniform_breaks, startendscenes, centralscenes, max_chunk_size, expdir, data_dir, model_name, model_prec, n_beams, n_dbeams, resumm_scenes=False, do_save_new_scenes=False, is_test=False, verbose=False):
        assert not (centralscenes and startendscenes)
        assert isinstance(expdir,str)
        self.device = device
        self.n_beams = n_beams
        self.n_dbeams = n_dbeams
        self.model_prec = model_prec
        self.model_name = self.dmodel_name = model_name
        self.verbose = verbose
        if self.device == 'cpu':
            self.dmodel = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            self.dmodel = load_peft_model(self.model_name, chkpt_path=None, precision=self.model_prec)
        #else:
            #self.dmodel = AutoModelForCausalLM.from_pretrained(self.dmodel_name).to(device)
        self.dtokenizer = AutoTokenizer.from_pretrained(self.dmodel_name, padding_side='left')
        self.dtokenizer.pad_token_id = self.dtokenizer.eos_token_id
        if self.model_name == self.dmodel_name:
            self.model = self.dmodel # avoid loading twice
            self.tokenizer = self.dtokenizer # avoid loading twice
        else: # this doesn't happen anymore, but may again as some baseline
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
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
        self.max_chunk_size = min(self.tokenizer.model_max_length, self.model.config.max_position_embeddings)
        self.dmax_chunk_size = min(self.dtokenizer.model_max_length, self.dmodel.config.max_position_embeddings)
        self.resumm_scenes = resumm_scenes
        self.do_save_new_scenes = do_save_new_scenes
        self.is_test = is_test
        self.bs = bs
        self.dbs = dbs
        self.fn = get_fn(caps, self.scene_order, self.uniform_breaks, self.startendscenes, self.centralscenes, self.is_test)
        self.desired_summ_len = 635 # mean in Moviesumm testset

    def pad_batch(self,batch,tokenizer):
        N=max([len(c) for c in batch])
        attention_mask = torch.stack([torch.cat([torch.zeros(N-len(c)),torch.ones(len(c))]) for c in batch], dim=0).to(self.device)
        padded = [[tokenizer.eos_token_id]*(N-len(b))+b for b in batch]
        padded = torch.tensor(padded).to(self.device)
        assert padded.shape == attention_mask.shape
        assert (padded[~attention_mask.bool()]==tokenizer.pad_token_id).all()
        return padded, attention_mask

    def summ_scenes(self, vidname, infer_splits):
        ep = episode_from_name(vidname, infer_splits)
        scenes = ['']*len(ep.scenes) if self.caps_only else ep.scenes
        if len(scenes) == 1:
            print(f'no scene breaks for {vidname}')
            breakpoint()
        if self.caps == 'nocaptions':
            caps = ['']*len(scenes)
        else: # prepend vid caps to the scene summ
            with open(join(self.data_dir,f'postprocessed-video-captions/{vidname}/{self.caps}_procced_scene_caps.json')) as f:
                caps_data = json.load(f)
            cdd = {c['scene_id']:c['with_names'] for c in caps_data}
            caps = [cdd.get(f'{vidname}s{i}','') for i in range(len(scenes))]
            assert len(caps)==len(scenes)
        assert all('talking' not in x for x in caps)
        if self.uniform_breaks:
            transcript_wo_scene_marks = '\n'.join([x for x in ep.transcript if x!='[SCENE_BREAK]'])
            combined_scenes = chunkify(transcript_wo_scene_marks, self.dmax_chunk_size)
            combined_caps = caps
        else:
            combined_scenes = scenes
            combined_caps = caps

        if self.caps_only:
            return combined_caps
        scene_summarize_prompt = lambda i,s: f'Here is the dialogue from scene {i} of the movie {titleify(vidname)}. Please give a few bullet points of its main events. Don\'t include information from outside this scene. Do not answer in progressive aspect, i.e., don\'t use -ing verbs or "is being".\n{s}\nIn this scene, '
        global_contraction_rate = sum(len(s.split()) for s in combined_scenes) / self.desired_summ_len
        print(len(combined_scenes))
        #combined_scenes = [s for s in combined_scenes if len(s.removeprefix(scene_summarize_prompt).split())/global_contraction_rate**.5 > 10]
        combined_scenes = [s for s in combined_scenes if len(s.split())/global_contraction_rate**.5 > 10][:ARGS.exclude_scenes_after]
        print(len(combined_scenes))
        combined_scenes = [scene_summarize_prompt(i,c) for i,c in enumerate(combined_scenes)]
        chunk_list = [chunkify(s, self.dmax_chunk_size) for s in combined_scenes]
        chunks = sum(chunk_list,[])
        assert (chunks==combined_scenes) or not self.uniform_breaks
        #avg_scene_summ_len = self.dmax_chunk_size//len(chunks)
        tok_chunks = [self.dtokenizer(c)['input_ids'] for c in chunks]
        sort_idxs = np.argsort([len(x) for x in tok_chunks])
        reversed_sort_idxs = np.argsort(sort_idxs)
        sorted_tok_chunks = [tok_chunks[i] for i in sort_idxs]
        short_chunk_summs = []
        remaining_chunks = sorted_tok_chunks
        assert all([sorted_tok_chunks[reversed_sort_idxs[i]]==c for i,c in enumerate(tok_chunks)])
        N = ceil(len(remaining_chunks)/self.dbs)
        remaining_chunk_summs = []
        #n_toks_in_whole_movie = len(self.dtokenizer(' '.join(combined_scenes)).input_ids)
        for i in range(N):
            padded, attn = self.pad_batch(remaining_chunks[i*self.dbs:(i+1)*self.dbs],self.dtokenizer)
            #n_toks_in_scene = len(remaining_chunks[i*self.dbs])
            #expected_len = self.dmax_chunk_size * n_toks_in_scene/n_toks_in_whole_movie
            mean_scene_len = (sum([len(c) for c in remaining_chunks[i*self.dbs:(i+1)*self.dbs]]) / self.dbs) - len(self.dtokenizer(scene_summarize_prompt(0,'')).input_ids)
            expected_len = mean_scene_len / global_contraction_rate**.5
            min_len = int(expected_len) - ARGS.scene_min_minus
            max_len = min_len + 10
            if padded.shape[1] > self.dmax_chunk_size:
                padded = padded[:,:self.dmax_chunk_size]
                attn = attn[:,:self.dmax_chunk_size]
            summ_tokens = self.dmodel.generate(padded, attention_mask=attn, min_new_tokens=min_len, max_new_tokens=max_len, num_beams=self.n_dbeams)
            summ_tokens = summ_tokens[:, padded.shape[1]:]
            assert summ_tokens.shape[1] <= max_len
            summ = self.dtokenizer.batch_decode(summ_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            assert attn[-1].all()
            if attn[0].argmax()==0:
                assert attn.all()
            remaining_chunk_summs += summ
            if self.verbose:
                print(i,summ)
            #self.dtokenizer.decode(self.dmodel.generate(torch.tensor([self.dtokenizer("Give me a simple sentence with the word 'long' in it").input_ids], device='cuda'), min_length=100)[0])
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
        ss = ['' if x=='' else f'In scene {i},{x[0].lower()}{x[1:]}' for i,x in enumerate(desplit)]
        caps = ['' if sc=='' or 'UNK' in sc or 'at the camera' in sc else f'On camera, {sc}' for i,sc in enumerate(combined_caps)]
        ss_with_caps = [x+sc for sc,x in zip(ss, caps)]
        breakpoint()
        if self.caps == 'nocaptions':
            assert self.tokenizer.model_max_length + 15*len(chunks) >= len(self.dtokenizer(''.join(ss_with_caps))[0])
        return ss_with_caps

    def get_scene_summs(self, epname, infer_splits):
        epname_ = f'{epname}-inferred' if infer_splits else epname
        scene_summ_dir = join(self.data_dir, 'scene_summs')
        maybe_scene_summ_path = join(scene_summ_dir, f'{epname_}_{self.fn}.txt')
        if os.path.exists(maybe_scene_summ_path) and not self.resumm_scenes:
            with open(maybe_scene_summ_path) as f:
                ss = [x.strip() for x in f.readlines()]
        else:
            ss = self.summ_scenes(epname, infer_splits)
            if self.do_save_new_scenes:
                check_dir(scene_summ_dir)
                with open(maybe_scene_summ_path,'w') as f:
                    f.write('\n'.join(ss))
        return ss

    def summarize_from_vidname(self, vidname):
        with torch.no_grad():
            scene_summs = self.get_scene_summs(vidname, infer_splits=False)
            return self.summarize_scene_summs('\n'.join(scene_summs), vidname)

    def summarize_scene_summs(self, concatted_scene_summs, vidname):
        if self.caps_only:
            min_chunk_len = 80
            max_chunk_len = 100
        else:
            min_chunk_len = int((self.desired_summ_len-ARGS.min_minus)*4/3)
            #min_chunk_len = 80
            max_chunk_len = min_chunk_len + 60
        summarize_prompt = f'Here is a sequence of summaries of each scene of the movie {titleify(vidname)}. {concatted_scene_summs}\nCombine them into a plot synopsis of no more than {max_chunk_len} words. Do not write the summary in progressive aspect, i.e., don\'t use -ing verbs or "is being". Be sure to include information from all scenes, especially those at the end. Focus only on the plot events, no analysis or discussion of themes and characters.'
        chunks = chunkify(summarize_prompt, self.max_chunk_size)
        assert len(chunks) == 1
        tok_chunks = [self.tokenizer(c)['input_ids'] for c in chunks]
        pbatch, attn = self.pad_batch(tok_chunks,self.tokenizer)
        summarize_prompt = f'{concatted_scene_summs}\nCombine them into a single summary for the entire movie. '
        meta_chunk_toks = self.model.generate(pbatch, attention_mask=attn, min_new_tokens=min_chunk_len, max_new_tokens=max_chunk_len, num_beams=self.n_beams)
        meta_chunk_toks = meta_chunk_toks[:, pbatch.shape[1]:]
        text_mcss = self.tokenizer.batch_decode(meta_chunk_toks,skip_special_tokens=True)
        text_mcss = [drop_trailing_halfsent(tmcs) for tmcs in text_mcss]
        final_summ = ' '.join(text_mcss)
        print(final_summ)
        breakpoint()
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
        import pandas as pd
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

        assert all([os.path.isdir(join(self.data_dir,f'postprocessed-video-captions/{x}')) for x in epnames])
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
            self.model = AutoModelForCausalLM.from_pretrained(best_chkpt).to(self.device)
        test_rouges = self.inference_epoch(self.n_epochs, testset, 'test')
        return test_rouges, alltime_best_rouges, all_rouges

def drop_trailing_halfsent(s):
    s = s.replace('Dr.','XXX')
    s = '. '.join(x for x in s.split('. ')[:-1])
    s = s.replace('XXX', 'Dr.')
    return s

def load_peft_model(base_model_name_or_path, chkpt_path, precision):
    print('loading model from', base_model_name_or_path)
    if precision==32:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path).cuda()
    elif precision==8:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    elif precision==4:
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    else:
        assert precision==2
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_4bit=True))
    #if chkpt_path is None:
    #    print('no peft chkpt to update from')
    #    model = prepare_model_for_kbit_training(model)
    #    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    #    model = get_peft_model(model,lora_config)
    #else:
    #    print('updating model with peft chkpt from', chkpt_path)
    #    config = PeftConfig.from_pretrained(chkpt_path)
    #    assert config.base_model_name_or_path==base_model_name_or_path
    #    model.enable_input_require_grads()
    #    model = PeftModel.from_pretrained(model, chkpt_path, is_trainable=True)
    #    assert any([x.requires_grad for x in model.parameters()])
    model.eval()
    return model

def titleify(vn):
     title_lower = vn.split('_')[0].replace('-', ' ')
     title = ' '.join([w[0].upper()+w[1:] for w in title_lower.split()])
     return title

if __name__ == '__main__':
    import argparse

    #openai.api_key = "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"
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-beams', type=int, default=3)
    parser.add_argument('--n-dbeams','-n', type=int, default=3)
    parser.add_argument('--min-minus', type=int, default=30)
    parser.add_argument('--scene-min-minus', type=int, default=5)
    parser.add_argument('--summ-scenes-only', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--resumm-scenes', action='store_true')
    parser.add_argument('--caps', type=str, choices=['swinbert','kosmos','nocaptions'],default='nocaptions')
    parser.add_argument('--order', type=str, choices=['identity','optimal','rand'], default='identity')
    parser.add_argument('-t','--is-test', action='store_true')
    parser.add_argument('--recompute-scene-summs', action='store_true')
    parser.add_argument('--prec', type=int, default=32, choices=[32,8,4,2])
    parser.add_argument('--vidname', type=str, default='the-sixth-sense_1999')
    parser.add_argument('--dbs', type=int, default=8)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--exclude-scenes-after', type=int, default=99999)
    parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    ARGS = parser.parse_args()

    if ARGS.vidname == 'all':
        #with open('moviesumm_testset_names.txt') as f:
        #    official_names = f.read().split('\n')
        #with open('clean-vid-names-to-command-line-names.json') as f:
        #    clean2cl = json.load(f)
        #assert all(x in official_names for x in clean2cl.keys())
        #test_vidnames = list(clean2cl.values())
        test_vidnames, clean2cl = get_all_testnames()
    else:
        test_vidnames = [ARGS.vidname]
    #assert (ARGS.device=='cpu') == (ARGS.prec==32)
    llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
                'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                }
    summarizer_model = SoapSummer(
                device=ARGS.device,
                model_name=llm_dict[ARGS.model],
                model_prec=ARGS.prec,
                bs=ARGS.bs,
                n_dbeams=ARGS.n_beams,
                n_beams=ARGS.n_dbeams,
                dbs=ARGS.dbs,
                caps='kosmos',
                scene_order='identity',
                uniform_breaks=False,
                startendscenes=False,
                centralscenes=False,
                max_chunk_size=10000,
                expdir='experiments',
                data_dir='./data',
                resumm_scenes=ARGS.recompute_scene_summs,
                do_save_new_scenes=True,
                is_test=ARGS.is_test,
                verbose=ARGS.verbose,
                )

    nparams = sum(x.numel() for x in summarizer_model.model.parameters())
    print(f'Summarization model has {nparams} parameters')
    for vn in test_vidnames:
        concatted_scene_summs, final_summ = summarizer_model.summarize_from_vidname(vn)
        print(concatted_scene_summs)
        check_dir(f'experiments/{vn}')
        with open(f'experiments/{vn}/{vn}-summary.txt', 'w') as f:
            f.write(final_summ)
