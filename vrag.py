import os
from tqdm import tqdm
from run_iv import get_showseaseps
import pandas as pd
import copy
import numpy as np
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from utils import drop_trailing_halfsent
import torch
import argparse
import json
from natsort import natsorted
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
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
    for fn in natsorted(os.listdir(f'rag-caches/{vid_subpath}/{split_name}/names')):
        with open(f'rag-caches/{vid_subpath}/{split_name}/names/{fn}') as f:
            names_in_scenes.append(f.read().split('\n'))
    for fn in natsorted(os.listdir(f'rag-caches/{vid_subpath}/{split_name}/scene_texts')):
        with open(f'rag-caches/{vid_subpath}/{split_name}/scene_texts/{fn}') as f:
            scenes.append(f.read().split('\n'))

    if os.path.exists(lava_out_fp:=f'lava-outputs/{vid_subpath}/{split_name}/all.json'):
        with open(lava_out_fp) as f:
            viz_texts = json.load(f)
    else:
        print(f'no lava out file at {lava_out_fp}')
        viz_texts = {f'scene{i}':'' for i in range(len(scenes))}
    return names_in_scenes, scenes, viz_texts

def answer_qs(show_name, season, episode, model, ep_qs):
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'

    if ARGS.splits == 'none':
        _, scenes, viz_texts = get_texts('ours', vid_subpath)
        scene_text = '[SCENE_BREAK]'.join('\n'.join(l for l in s) for s in scenes)
        viz_scene_text = '\n'.join(drop_trailing_halfsent(s) for s in viz_texts.values())
    else:
        names_in_scenes, scenes, viz_texts = get_texts(ARGS.splits, vid_subpath)

        scene_text_feats = [torch.load(f'rag-caches/{vid_subpath}/{ARGS.splits}/text_feats/{fn}').to(device) for fn in os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/text_feats/')][:5000]
        if len(scene_text_feats)==0:
            print(f'{show_name} {season} {episode}: empty scene texts')
            return 0, 0
        scene_text_feats = torch.stack([torch.zeros(512, device=device) if len(x)==0 else x.mean(axis=0) for x in scene_text_feats])
        scene_vision_feats = torch.cat([torch.load(f'rag-caches/{vid_subpath}/{ARGS.splits}/vid_feats/{fn}') for fn in os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/vid_feats')])
        if scene_vision_feats.shape[0] == scene_text_feats.shape[0]:
            scene_feats = (scene_vision_feats + scene_text_feats) / 2
        else:
            print(f'{show_name} {season} {episode}: vision feats have {len(scene_vision_feats)} scenes while text feats have {len(scene_text_feats)}, names is {len(names_in_scenes)}, scenes is {len(scenes)}')
            scene_feats = scene_text_feats

    if ARGS.test_loading:
        return 0,0
    n_correct = 0
    if ARGS.splits == 'none':
        recurring_prompt_prefix = f'Answer the given question based on the following text:\n{viz_scene_text}\n{scene_text}\n'[:5000]
        incr = 1000
        for n_tries in range(len(recurring_prompt_prefix)//incr):
            try:
                prompt_cache = DynamicCache()
                inputs = tokenizer(recurring_prompt_prefix, return_tensors="pt").to(device)
                prompt_cache = model(**inputs, past_key_values = prompt_cache).past_key_values # this is the common prompt cached
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(e)
                recurring_prompt_prefix = recurring_prompt_prefix[incr:]
                print(f'OOM, reducing min,max to {len(recurring_prompt_prefix)}chars')

    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        if ARGS.splits != 'none':
            qvec = torch.load(f'rag-caches/{vid_subpath}/qfeats/{i}.pt')
            sims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_feats])
            names_in_q = [w.replace('Anabelle', 'Annabelle') for w in word_tokenize(qdict['q']) if w in all_names]
            names_match = torch.tensor([all(n in ns for n in names_in_q) for ns in names_in_scenes]).float()
            sims[~names_match.bool()] -= torch.inf

            scene_text = '\n'.join(scenes[sims.argmax()])
            viz_scene_text = drop_trailing_halfsent(viz_texts[f'scene{sims.argmax()}'])
        options = '\n'.join(k[1] + ': ' + qdict[k] for k in ('a0', 'a1', 'a2', 'a3', 'a4'))
        question_part = f'Question: {qsent}\nSelect the answer from the following options:\n{options}\nJust give the number of the answer. Your answer should only be a number from 1-4, no punctuation or whitespace.'
        if ARGS.splits == 'none':
            prompt = recurring_prompt_prefix + question_part
            new_inputs = tokenizer(prompt, return_tensors="pt").to(device)
            #past_key_values = copy.deepcopy(prompt_cache)
            past_key_values = pc = DynamicCache()
            pc.key_cache = [t.clone() for t in prompt_cache.key_cache]
            pc.value_cache = [t.clone() for t in prompt_cache.value_cache]
            with torch.inference_mode():
                output = model.generate(**new_inputs, past_key_values=past_key_values, min_new_tokens=1, max_new_tokens=1, num_beams=1, output_scores=True, return_dict_in_generate=True)
        else:
            prompt = f'Answer the given question based on the following text:\n{viz_scene_text}\n{scene_text}\n{question_part}'
            tok_ids = torch.tensor([tokenizer(prompt).input_ids]).to(device)
            with torch.inference_mode():
               output = model.generate(tok_ids, attention_mask=torch.ones_like(tok_ids), min_new_tokens=1, max_new_tokens=1, num_beams=1, output_scores=True, return_dict_in_generate=True)
               #output = model(tok_ids)

        #ans_tokens = ans_tokens[0,tok_ids.shape[1]:]
        ans_token = output.sequences[0,-1:]
        ans = tokenizer.decode(ans_token, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        #ans = max(range(5), key=lambda i: output.scores[0,-1,tokenizer.encode(str(i), add_special_tokens=False)[0]].item())
        ans = max(range(5), key=lambda i: output.scores[0][-1,tokenizer.encode(str('a'), add_special_tokens=False)[0]].item())
        if ARGS.verbose:
            print(prompt, qdict['answer_idx'])
            print(f'pred: {ans} gt: {qdict["answer_idx"]}')
        if ans==qdict['answer_idx']:
            n_correct += 1
    n = len(ep_qs["questions"])
    print(f'vqa acc: {n_correct}/{n} = {n_correct/n:.5f}')
    return n_correct, n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute-scene-texts', action='store_true')
    parser.add_argument('--test-loading', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif', 'none'])
    parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dud', action='store_true')
    ARGS = parser.parse_args()


    from hierarchical_summarizer import load_peft_model
    llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
                'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                }
    model_name = llm_dict[ARGS.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if ARGS.cpu:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        device = 'cpu'
    else:
        model = load_peft_model(model_name, None, ARGS.prec)
        device = 'cuda'
    model.eval()

    tot_n_correct, tot = 0, 0
    all_scores = []

    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)
    all_scores = []
    for show_name, seas, ep in tqdm(showseaseps):
        season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{seas}']
        if f'episode_{ep}' not in season_qs.keys():
            print(f'Episode_{ep} not in season_{seas} keys')
            continue
        ep_qs = season_qs[f'episode_{ep}']
        new_correct, new_tot = answer_qs(show_name, seas, ep, model, ep_qs)
        all_scores.append([show_name, seas, ep, model, ep_qs, new_correct, new_tot, new_correct/new_tot])
    df = pd.DataFrame(all_scores, columns = ['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    print(df.mean(axis=0))
    os.makedirs(f'tvqa-results/{ARGS.splits}', exist_ok=True)
    df.to_csv(f'tvqa-results/{ARGS.splits}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv')
