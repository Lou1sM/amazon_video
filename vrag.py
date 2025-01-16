import os
from utils import drop_trailing_halfsent
import shutil
import re
from os.path import join
import numpy as np
import pandas as pd
import torch
import argparse
import json
from natsort import natsorted
from transformers import AutoModelForCausalLM
from nltk.corpus import names
from nltk.tokenize import word_tokenize
import nltk
nltk.download('names')
male_names = names.words('male.txt')
female_names = names.words('female.txt')
all_names = [n for n in male_names + female_names]


with open('tvqa-long-annotations_tvqa_val_edited.json') as f:
    full_dset_qs = json.load(f)

with open('tvqa_preprocessed_subtitles.json') as f:
    full_dset_subs = json.load(f)

with open('tvqa-splits.json') as f:
    tvqa_splits = json.load(f)

show_name_dict = {
                  'friends':'Friends',
                  'house': 'House M.D.',
                  'met': 'How I Met Your Mother',
                  'bbt': 'The Big Bang Theory',
                  'castle': 'Castle',
                  'grey': "Grey's Anatomy",
                  }

def answer_qs(show_name, season, episode, model, ep_qs):
    #print(show_name, season, episode)
    #return 0, 0
    #ep_id = f'{show_name}_s{season:02}e{episode:02}'
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'
    #kf_times = np.load(f'data/ffmpeg-keyframes/{vid_subpath}/frametimes.npy')

    ep_qs = full_dset_qs[show_name[0].upper()+show_name[1:]][f'season_{season}'][f'episode_{episode}']
    #dset_subs = [x for x in full_dset_subs if x['vid_name'].startswith(f'{show_name}_s{int(season):02}e{int(episode):02}')]

    #scene_split_points = np.load(f'data/ffmpeg-keyframes-by-scene/{vid_subpath}/scenesplit_timepoints.npy')
    #scene_split_points = np.array([0] + list(scene_split_points) + [float(kf_times.max())])

    #if not os.path.exists(transcript_fp:=f'data/tvqa-transcripts/{ep_id}.json'):
        #return 0, 0
    #with open(transcript_fp) as f:
        #tlines_with_breaks = json.load(f)['Transcript']

    #tlines = [x for x in tlines_with_breaks if x!='[SCENE_BREAK]']
    for d in ('names', 'scene_texts', 'text_feats'):
        os.makedirs(f'rag-caches/{vid_subpath}/{d}', exist_ok=True)
    scene_text_feats = [torch.load(f'rag-caches/{vid_subpath}/text_feats/{fn}') for fn in os.listdir(f'rag-caches/{vid_subpath}/text_feats/')]
    if len(scene_text_feats)==0:
        print(f'{show_name} {season} {episode}: empty scene texts')
        return 0, 0
    scene_text_feats = torch.stack([torch.zeros(512) if len(x)==0 else x.mean(axis=0) for x in scene_text_feats])
    scenes = []
    names_in_scenes = []
    viz_texts = []
    for fn in natsorted(os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/names')):
        with open(f'rag-caches/{vid_subpath}/names/{fn}') as f:
            names_in_scenes.append(f.read().split('\n'))
    for fn in natsorted(os.listdir(f'rag-caches/{vid_subpath}/{ARGS.splits}/scene_texts')):
        with open(f'rag-caches/{vid_subpath}/scene_texts/{fn}') as f:
            scenes.append(f.read().split('\n'))

    if os.path.exists(lava_out_fp:=f'data/lava-outputs/{vid_subpath}/{ARGS.splits}/all.json'):
        with open(lava_out_fp) as f:
            viz_texts = json.load(f)
    else:
        print(f'no lava out file at {lava_out_fp}')
        viz_texts = {f'scene{i}':'' for i in range(len(scenes))}

    #clip_timepoints = np.cumsum([0] + [max(x['end'] for x in clip['sub']) for clip in sorted(dset_subs, key=lambda x:x['vid_name'])])

    scene_vision_feats = torch.cat([torch.load(f'data/internvid-feats/{vid_subpath}/{ARGS.splits}/{fn}') for fn in os.listdir(f'data/internvid-feats/{vid_subpath}/{ARGS.splits}')[:-1]])
    #scene_text_feats = torch.stack(scene_text_feats)
    if scene_vision_feats.shape[0] == scene_text_feats.shape[0]:
        scene_feats = (scene_vision_feats + scene_text_feats) / 2
    else:
        print(f'{show_name} {season} {episode}: vision feats have {len(scene_vision_feats)} scenes while text feats have {len(scene_text_feats)}, names is {len(names_in_scenes)}, scenes is {len(scenes)}')
        scene_feats = scene_text_feats
    #scene_feats = scene_vision_feats

    if ARGS.test_loading:
        return 0,0
    n_correct = 0
    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        #qvec = text_model.get_txt_feat(qsent:=qdict['q'])
        qvec = torch.load(f'rag-caches/{vid_subpath}/qfeats/{i}.pt')
        #vsims = (scene_vision_feats @ qvec.T).squeeze()
        #tsims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_feats])
        #sims = tsims + vsims
        sims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_feats])
        names_in_q = [w.replace('Anabelle', 'Annabelle') for w in word_tokenize(qdict['q']) if w in all_names]
        names_match = torch.tensor([all(n in ns for n in names_in_q) for ns in names_in_scenes]).float()
        #sims = sims*names_match
        sims[~names_match.bool()] -= torch.inf

        #pred_scene_idxs = sims.topk(1).indices
        #scene_text = '[SCENE_BREAK]'.join('\n'.join(scenes[i]) for i in pred_scene_idxs)
        scene_text = '\n'.join(scenes[sims.argmax()])
        viz_scene_text = drop_trailing_halfsent(viz_texts[f'scene{sims.argmax()}'])
        options = '\n'.join(k[1] + ': ' + qdict[k] for k in ('a0', 'a1', 'a2', 'a3', 'a4'))
        prompt = f'Answer the given question based on the following text:\n{viz_scene_text}\n{scene_text}\nQuestion: {qsent}\nSelect the answer from the following options:\n{options}\nJust give the number of the answer. Your answer should only be a number from 1-4, no punctuation or whitespace.'
        tok_ids = torch.tensor([tokenizer(prompt).input_ids]).to(device)
        if ARGS.dud:
            ans = 0
        else:
            with torch.no_grad():
               ans_tokens = model.generate(tok_ids, min_new_tokens=1, max_new_tokens=1, num_beams=1)
               output = model(tok_ids)
            ans_tokens = ans_tokens[0,tok_ids.shape[1]:]
            ans = tokenizer.decode(ans_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ans = max(range(5), key=lambda i: output.logits[0,-1,tokenizer.encode(str(i), add_special_tokens=False)[0]].item())
        print(prompt, qdict['answer_idx'])
        print(f'pred: {ans} gt: {qdict["answer_idx"]}')
        if ans==qdict['answer_idx']:
            n_correct += 1
    n = len(ep_qs["questions"])
    print(f'vqa acc: {n_correct/n}')
    return n_correct, n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dset', '-d', type=str, default='tvqa')
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    #parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--recompute-scene-texts', action='store_true')
    parser.add_argument('--test-loading', action='store_true')
    parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif'])
    parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dud', action='store_true')
    ARGS = parser.parse_args()


    from hierarchical_summarizer import load_peft_model
    from tqdm import tqdm
    from transformers import AutoTokenizer
    llm_dict = {'llama3-tiny': 'llamafactory/tiny-random-Llama-3',
                'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                }
    model_name = llm_dict[ARGS.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if ARGS.cpu:
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
        device = 'cpu'
    else:
        model = load_peft_model(model_name, None, ARGS.prec)
        device = 'cuda'
    model.eval()

    tot_n_correct, tot = 0, 0
    def qa_season(seas_num):
        season_qs = full_dset_qs[show_name_dict[ARGS.show_name]][f'season_{seas_num}']
        global tot_n_correct
        scores_by_ep = {}
        for ep in os.listdir(f'data/ffmpeg-keyframes-by-scene/tvqa/{ARGS.show_name}/season_{seas_num}'):
            if ep not in season_qs.keys():
                print(f'Episode_{ep} not in season_{seas_num} keys')
                continue
            else:
                ep_qs = season_qs[ep]
            ep_num = ep.removeprefix('episode_')
            new_correct, new_tot = answer_qs(ARGS.show_name, seas_num, ep_num, model, ep_qs)
            tot_n_correct += new_correct
            new_tot += tot
            scores_by_ep[ep_num] = {'n_correct': new_correct, 'tot': new_tot}
        return scores_by_ep

    if ARGS.season == -1:
        seasons = sorted([x.removeprefix('season_') for x in os.listdir(f'data/full-videos/tvqa/{ARGS.show_name}')])
        scores = {}
        for s in seasons:
            seas_scores = qa_season(s)
            scores[f'season_{s}'] = seas_scores
    else:
        seas_scores = qa_season(ARGS.season)
        scores = {f'season_{ARGS.season}': seas_scores}
    scores['tot_n_correct'] = tot_n_correct
    scores['tot'] = tot
    with open(f'{ARGS.show_name}_{ARGS.season}-tvqa-results.json', 'w') as f:
        json.dump(scores, f)

