import os
import numpy as np
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
from utils import drop_trailing_halfsent
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

show_name_dict = {
                  'friends':'Friends',
                  'house': 'House M.D.',
                  'met': 'How I Met Your Mother',
                  'bbt': 'The Big Bang Theory',
                  'castle': 'Castle',
                  'grey': "Grey's Anatomy",
                  }

def answer_qs(show_name, season, episode, model, ep_qs):
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'

    ep_qs = full_dset_qs[show_name[0].upper()+show_name[1:]][f'season_{season}'][f'episode_{episode}']
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

    scene_vision_feats = torch.cat([torch.load(f'data/internvid-feats/{vid_subpath}/{ARGS.splits}/{fn}') for fn in os.listdir(f'data/internvid-feats/{vid_subpath}/{ARGS.splits}')[:-1]])
    if scene_vision_feats.shape[0] == scene_text_feats.shape[0]:
        scene_feats = (scene_vision_feats + scene_text_feats) / 2
    else:
        print(f'{show_name} {season} {episode}: vision feats have {len(scene_vision_feats)} scenes while text feats have {len(scene_text_feats)}, names is {len(names_in_scenes)}, scenes is {len(scenes)}')
        scene_feats = scene_text_feats

    if ARGS.test_loading:
        return 0,0
    n_correct = 0
    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        qvec = torch.load(f'rag-caches/{vid_subpath}/qfeats/{i}.pt')
        sims = torch.tensor([(ts @ qvec.T).max(axis=0).values for ts in scene_feats])
        names_in_q = [w.replace('Anabelle', 'Annabelle') for w in word_tokenize(qdict['q']) if w in all_names]
        names_match = torch.tensor([all(n in ns for n in names_in_q) for ns in names_in_scenes]).float()
        sims[~names_match.bool()] -= torch.inf

        scene_text = '\n'.join(scenes[sims.argmax()])
        viz_scene_text = drop_trailing_halfsent(viz_texts[f'scene{sims.argmax()}'])
        options = '\n'.join(k[1] + ': ' + qdict[k] for k in ('a0', 'a1', 'a2', 'a3', 'a4'))
        prompt = f'Answer the given question based on the following text:\n{viz_scene_text}\n{scene_text}\nQuestion: {qsent}\nSelect the answer from the following options:\n{options}\nJust give the number of the answer. Your answer should only be a number from 1-4, no punctuation or whitespace.'
        tok_ids = torch.tensor([tokenizer(prompt).input_ids]).to(device)
        if ARGS.dud:
            ans = 0
        else:
            with torch.no_grad():
               ans_tokens = model.generate(tok_ids, attention_mask=torch.ones_like(tok_ids), min_new_tokens=1, max_new_tokens=1, num_beams=1)
               output = model(tok_ids)
            ans_tokens = ans_tokens[0,tok_ids.shape[1]:]
            ans = tokenizer.decode(ans_tokens,skip_special_tokens=True, clean_up_tokenization_spaces=True)
            ans = max(range(5), key=lambda i: output.logits[0,-1,tokenizer.encode(str(i), add_special_tokens=False)[0]].item())
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
    parser.add_argument('--recompute-scene-texts', action='store_true')
    parser.add_argument('--test-loading', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument("--splits", type=str, default='ours', choices=['ours', 'psd', 'unif'])
    parser.add_argument('--model', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dud', action='store_true')
    ARGS = parser.parse_args()


    from hierarchical_summarizer import load_peft_model
    from transformers import AutoTokenizer
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
    def qa_season(seas_num):
        season_qs = full_dset_qs[show_name_dict[ARGS.show_name]][f'season_{seas_num}']
        global tot_n_correct
        global tot
        scores_by_ep = {}
        for ep in os.listdir(f'rag-caches/tvqa/{ARGS.show_name}/season_{seas_num}'):
            if ep not in season_qs.keys():
                print(f'Episode_{ep} not in season_{seas_num} keys')
                continue
            else:
                ep_qs = season_qs[ep]
            ep_num = ep.removeprefix('episode_')
            new_correct, new_tot = answer_qs(ARGS.show_name, seas_num, ep_num, model, ep_qs)
            tot_n_correct += new_correct
            tot += new_tot
            scores_by_ep[ep_num] = {'n_correct': new_correct, 'tot': new_tot}
            if new_tot==0:
                print(888)
            else:
                all_scores.append(new_correct/new_tot)
        return scores_by_ep

    if ARGS.season == -1:
        seasons = sorted([x.removeprefix('season_') for x in os.listdir(f'rag-caches/tvqa/{ARGS.show_name}')])
        scores = {}
        for s in seasons:
            seas_scores = qa_season(s)
            scores[f'season_{s}'] = seas_scores
    else:
        seas_scores = qa_season(ARGS.season)
        scores = {f'season_{ARGS.season}': seas_scores}
    scores['tot_n_correct'] = tot_n_correct
    scores['tot'] = tot
    print(f'macro: {np.array(all_scores).mean():.4f}, micro: {scores["tot_n_correct"]/scores["tot"]:.4f}')
    os.makedirs(f'tvqa-results/{ARGS.splits}', exist_ok=True)
    with open(f'tvqa-results/{ARGS.splits}/{ARGS.show_name}_{ARGS.season}-tvqa-results.json', 'w') as f:
        json.dump(scores, f)

