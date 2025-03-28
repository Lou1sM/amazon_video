import os
from os.path import join
from tqdm import tqdm
import pandas as pd
from time import time
import copy
import numpy as np
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from utils import drop_trailing_halfsent
import torch
import argparse
import json
from natsort import natsorted
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,  BitsAndBytesConfig
from nltk.corpus import names
from nltk.tokenize import word_tokenize
import nltk
nltk.download('names')
male_names = names.words('male.txt')
female_names = names.words('female.txt')
all_names = [n for n in male_names + female_names]


with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

show_name_dict = {
                  'friends':'Friends',
                  'house': 'House M.D.',
                  'met': 'How I Met You Mother',
                  'bbt': 'The Big Bang Theory',
                  'castle': 'Castle',
                  'grey': "Grey's Anatomy",
                  }

def get_texts(split_name, vid_subpath):
    scenes = []
    names_in_scenes = []
    viz_texts = []
    for fn in natsorted(os.listdir(rag_caches_dir:=join(ARGS.rag_caches_prefix, 'rag-caches', vid_subpath, split_name, 'names'))):
        with open(join(rag_caches_dir, fn)) as f:
            names_in_scenes.append(f.read().split('\n'))
    for fn in natsorted(os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', vid_subpath, split_name, 'scene_texts'))):
        with open(f'rag-caches/{vid_subpath}/{split_name}/scene_texts/{fn}') as f:
            scenes.append(f.read())

    if os.path.exists(lava_out_dir:=join(ARGS.lava_outputs_prefix, 'lava-outputs', split_name, vid_subpath)):
        for fn in os.listdir(lava_out_dir):
            if fn=='all.json':
                continue
            assert '.' not in fn # shouldn't have ext
            with open(f'{lava_out_dir}/{fn}') as f:
                viz_texts.append(drop_trailing_halfsent(f.read()))
        if len(viz_texts)>len(scenes):
            #print(f'viz texts len {len(viz_texts)} but scenes len {len(scenes)} for {vid_subpath}, so cutting short')
            #viz_texts = {f'scene{i}':viz_texts[f'scene{i}'] for i in range(len(scenes))}
            viz_texts = viz_texts[:len(scenes)]
        if len(viz_texts) != len(scenes):
            breakpoint()
    else:
        print(f'no lava out files in {lava_out_dir}')
        viz_texts = {f'scene{i}':'' for i in range(len(scenes))}
    vl_texts = [scenes[i] + viz_texts[i] for i in range(len(scenes))]
    return vl_texts, names_in_scenes, scenes, viz_texts

def get_showseaseps(show_name_, seas_num_, ep_num_):
    showseaseps = []
    if show_name_=='all':
        show_names_to_compute = natsorted(os.listdir(f'rag-caches/tvqa/'))
        show_names_to_compute = [x for x in show_names_to_compute if x!='bbt']
    else:
        show_names_to_compute = [show_name_]
    for show_name in show_names_to_compute:
        if seas_num_ == -1:
            seass_to_compute = natsorted([int(fn[7:]) for fn in os.listdir(f'rag-caches/tvqa/{show_name}')])
        else:
            seass_to_compute = [seas_num_]

        for seas_num in seass_to_compute:
            if ep_num_ == -1:
                for fn in natsorted(os.listdir(f'rag-caches/tvqa/{show_name}/season_{seas_num}')):
                    ep_num = int(fn[8:].removesuffix('.mp4'))
                    showseaseps.append((show_name, seas_num, ep_num))
            else:
                showseaseps.append((show_name, seas_num, ep_num_))
    return showseaseps

class VQA():
    #def __init__(self):
        #self.ema_logits = np.array([1, -0.25, -0.25, -0.25, -0.25])

    def answer_qs(self, show_name, season, episode, model, ep_qs):
        dset_name = 'tvqa'
        vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'

        if ARGS.splits == 'none':
            vl_texts, _, scenes, viz_texts = get_texts('ours', vid_subpath)
            scene_text = '[SCENE_BREAK]'.join('\n'.join(l for l in s) for s in scenes)
            viz_scene_text = '\n'.join(viz_texts)
        else:
            vl_texts, names_in_scenes, scenes, viz_texts = get_texts(ARGS.splits, vid_subpath)

            scene_text_feats = []
            for fn in os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/text_feats/'):
                try:
                    stf = torch.load(f'rag-caches/{vid_subpath}/{ARGS.splits}/text_feats/{fn}').to(device)
                    stfm = torch.zeros(512, device=device) if len(stf)==0 else stf.mean(axis=0)
                except (RuntimeError, FileNotFoundError):
                    stfm = torch.zeros(512, device=device) if len(stf)==0 else stf.mean(axis=0)
                scene_text_feats.append(stfm)
            if len(scene_text_feats)==0:
                print(f'{show_name} {season} {episode}: empty scene texts')
                return 0, 0
            scene_text_feats = torch.stack(scene_text_feats)
            #scene_text_feats = torch.stack([torch.zeros(512, device=device) if len(x)==0 else x.mean(axis=0) for x in scene_text_feats])
            scene_vision_feats = torch.cat([torch.load(f'rag-caches/{vid_subpath}/{ARGS.splits}/vid_feats/{fn}').to(device) for fn in os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/vid_feats')])
            if scene_vision_feats.shape[0] == scene_text_feats.shape[0]:
                scene_feats = (scene_vision_feats + scene_text_feats) / 2
            else:
                print(f'{show_name} {season} {episode}: vision feats have {len(scene_vision_feats)} scenes while text feats have {len(scene_text_feats)}, names is {len(names_in_scenes)}, scenes is {len(scenes)}')
                scene_feats = scene_text_feats

        if ARGS.test_loading:
            return 0,0
        n_correct = 0
        if ARGS.splits == 'none':
            recurring_prompt_prefix = f'Answer the given question based on the following text:\n{scene_text}\n{viz_scene_text}\n'[:ARGS.prompt_prefix]
            starttime = time()
            initial_inputs = tokenizer(recurring_prompt_prefix, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**initial_inputs, use_cache=True)
            orig_past_key_values = outputs.past_key_values
            print(f'time for initial inputs: {time()-starttime:.3f}')


        for i, qdict in enumerate(ep_qs['questions']):
            qsent = qdict['q']
            if ARGS.splits != 'none':
                qvec = torch.load(f'rag-caches/{vid_subpath}/qfeats/{i}.pt').to(device)
                sims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_feats])
                names_in_q = [w.replace('Anabelle', 'Annabelle') for w in word_tokenize(qdict['q']) if w in all_names]
                names_match = torch.tensor([all(n in ns for n in names_in_q) for ns in names_in_scenes]).float()
                sims[~names_match.bool()] -= torch.inf

                retrieve_idx = sims.topk(ARGS.n_to_retrieve).indices
                scene_text = '\n'.join(vl_texts[i] for i in retrieve_idx)
            options = '\n'.join(k[1] + ': ' + qdict[k] for k in ('a0', 'a1', 'a2', 'a3', 'a4'))
            question_part = f'Question: {qsent}\nSelect the answer from the following options:\n{options}\nJust give the number of the answer. Your answer should only be a number from 0-4, no punctuation or whitespace.'
            if ARGS.splits == 'none':
                new_tokens = tokenizer(question_part, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                dynamic_pkv = tuple((k.clone().detach(), v.clone().detach()) for k, v in orig_past_key_values)
                print('now looping through question tokens')
                output = model(input_ids=new_tokens, past_key_values=dynamic_pkv, use_cache=True)
                ans_logits = output.logits
                prompt = recurring_prompt_prefix + question_part
            else:
                prompt = f'Answer the given question based on the following text:\n{scene_text}\n{question_part}'[:ARGS.prompt_prefix]
                tok_ids = torch.tensor([tokenizer(prompt).input_ids]).to(device)
                with torch.inference_mode():
                   output = model(tok_ids)
                   ans_logits = output.logits

            scores_by_answer = np.array([ans_logits[0, -1, tokenizer.encode(str(i), add_special_tokens=False)[0]].item() for i in range(5)])
            #self.ema_logits = (9*self.ema_logits + scores_by_answer) / 10
            #ans = (scores_by_answer - self.ema_logits).argmax()
            ans = scores_by_answer.argmax()
            if ARGS.verbose:
                print(prompt, qdict['answer_idx'])
                print(f'pred: {ans} gt: {qdict["answer_idx"]}')
                print('scores:', [ans_logits[0, -1, tokenizer.encode(str(i), add_special_tokens=False)[0]].item() for i in range(5)])
            if ans==qdict['answer_idx']:
                n_correct += 1
            print(ans, scores_by_answer)
        n = len(ep_qs["questions"])
        print(f'vqa acc: {n_correct}/{n} = {n_correct/n:.5f}')
        return n_correct, n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--test-loading', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif', 'none', 'GMM', 'scrl', 'bassl'])
    parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
    parser.add_argument('--n-to-retrieve', type=int, default=1)
    parser.add_argument('--prompt-prefix', type=int, default=5000)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dud', action='store_true')
    parser.add_argument("--rag-caches-prefix", type=str, default='.')
    parser.add_argument("--lava-outputs-prefix", type=str, default='.')
    ARGS = parser.parse_args()


    #from hierarchical_summarizer import load_peft_model
    llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
                'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                }
    model_name = llm_dict[ARGS.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    vqa = VQA()
    if ARGS.cpu:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        device = 'cpu'
    else:
        #model = load_peft_model(model_name, None, ARGS.prec)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=BitsAndBytesConfig(load_in_4bit=True), rope_scaling={"type": "linear","factor": 8.0})
        device = 'cuda'
    model.eval()

    tot_n_correct, tot = 0, 0
    all_scores = []

    os.makedirs(out_dir:=f'tvqa-results/{ARGS.splits}/{ARGS.model}-r{ARGS.n_to_retrieve}', exist_ok=True)
    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)
    print(showseaseps)
    all_scores = []
    for show_name, seas, ep in (pbar:=tqdm(showseaseps)):
        season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{seas}']
        if f'episode_{ep}' not in season_qs.keys():
            print(f'Episode_{ep} not in season_{seas} keys')
            continue
        if (show_name, seas, ep) == ('house', 4, 11): # no vid for some reason
            continue
        ep_qs = season_qs[f'episode_{ep}']
        cache_fp = os.path.join(out_dir, f'{show_name}_s{seas:01}e{ep:01}.json')
        if os.path.exists(cache_fp) and not ARGS.recompute:
            with open(cache_fp) as f:
                x = f.read().split()
            new_correct, new_tot = int(x[0]), int(x[1])
        else:
            new_correct, new_tot = vqa.answer_qs(show_name, seas, ep, model, ep_qs)
            with open(cache_fp, 'w') as f:
                f.write(f'{new_correct} {new_tot}')
        tot_n_correct += new_correct
        tot += new_tot
        all_scores.append([show_name, seas, ep, new_correct, new_tot, new_correct/new_tot])
        pbar.set_description(f'{show_name}-s{seas}e{ep}, running avg: {tot_n_correct}/{tot}={tot_n_correct/tot}')
    df = pd.DataFrame(all_scores, columns = ['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    print(df.drop('show', axis=1).mean(axis=0))
    df.to_csv(f'{out_dir}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv')
