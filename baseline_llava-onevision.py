import os
from PIL import Image
from os.path import join
from tqdm import tqdm
import pandas as pd
import logging
import torch
import argparse
import json
from natsort import natsorted
from decord import VideoReader, cpu
import numpy as np
import warnings
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import (get_model_name_from_path,
                           tokenizer_image_token,
                           KeywordsStoppingCriteria)
from llava.constants import (IMAGE_TOKEN_INDEX,
                           DEFAULT_IMAGE_TOKEN,
                           DEFAULT_IM_START_TOKEN,
                           DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
import copy

warnings.filterwarnings("ignore")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

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

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, video_time

def get_texts(split_name, vid_subpath):
    scenes = []
    for fn in natsorted(os.listdir(stexts_rag_caches_dir:=join(ARGS.rag_caches_prefix, 'rag-caches', split_name, vid_subpath, 'scene_texts'))):
        with open(join(stexts_rag_caches_dir, fn)) as f:
            scenes.append(f.read())
    return scenes

def get_showseaseps(show_name_, seas_num_, ep_num_):
    showseaseps = []
    if show_name_=='all':
        show_names_to_compute = natsorted(os.listdir(join(ARGS.rag_caches_prefix, 'rag-caches', 'ours', 'tvqa/')))
        show_names_to_compute = [x for x in show_names_to_compute if x!='bbt']
    else:
        show_names_to_compute = [show_name_]
    for show_name in show_names_to_compute:
        if seas_num_ == -1:
            seass_to_compute = natsorted([int(fn[7:]) for fn in os.listdir(join(ARGS.rag_caches_prefix, f'rag-caches', 'ours', 'tvqa', show_name))])
        else:
            seass_to_compute = [seas_num_]

        for seas_num in seass_to_compute:
            if ep_num_ == -1:
                for fn in natsorted(os.listdir(join(ARGS.rag_caches_prefix, f'rag-caches', 'ours', 'tvqa', show_name, f'season_{seas_num}'))):
                    ep_num = int(fn[8:].removesuffix('.mp4'))
                    showseaseps.append((show_name, seas_num, ep_num))
            else:
                showseaseps.append((show_name, seas_num, ep_num_))
    return showseaseps

def answer_qs(show_name, season, episode, model, processor, tokenizer, ep_qs):
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'
    scenes = get_texts('ours', vid_subpath)
    scene_text = '[SCENE_BREAK]'.join('\n'.join(l for l in s) for s in scenes)

    video_path = join(ARGS.rag_caches_prefix, f'data/full-videos', vid_subpath + '.mp4')
    max_frames_num = 8  # Reduced for memory efficiency

    try:
        video_frames, video_time = load_video(video_path, max_frames_num, force_sample=True)
        video = processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].to(ARGS.device)
        video = [video]
    except Exception as e:
        print(f"Error loading video: {e}")
        return 0, len(ep_qs["questions"])

    n_correct = 0
    conv_template = "qwen_1_5"

    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        options = '\n'.join(f"{idx}: {qdict[f'a{idx}']}" for idx in range(5))

        time_instruction = f"The video lasts for {video_time:.2f} seconds. Context: {scene_text}"
        question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\nQuestion: {qsent}\nOptions:\n{options}\nAnswer with just a number (0-4)."

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(ARGS.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids,images=video.half(),modalities=["video"],do_sample=False,temperature=0,max_new_tokens=1,)

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        try:
            ans = int(output.strip()[0])
            if 0 <= ans <= 4:
                if ans == qdict['answer_idx']:
                    n_correct += 1
                if ARGS.verbose:
                    print(f"Question: {qsent}")
                    print(f"Options: {options}")
                    print(f"Model output: {output}")
                    print(f"Predicted: {ans}, Correct: {qdict['answer_idx']}\n")
        except (ValueError, IndexError):
            pass

    n = len(ep_qs["questions"])
    print(f'VQA accuracy: {n_correct}/{n} = {n_correct/n:.5f}')
    breakpoint()
    return n_correct, n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--rag-caches-prefix", type=str, default='.')
    ARGS = parser.parse_args()

    ARGS.device = 'cpu' if ARGS.cpu else 'cuda'

    # Initialize LLaVA model
    pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
    #model_name = "llava_qwen"
    model_name = "LLaVA-Video-7B-Qwen2"
    #tokenizer, model, processor, _ = load_pretrained_model(
    #    pretrained,
    #    None,
    #    model_name,
    #    #torch_dtype='bfloat16',
    #    device_map="auto"
    #)
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_mode"] = 'average'
    overwrite_config["mm_spatial_pool_stride"] = 4
    overwrite_config["mm_newline_position"] = 'no_token'
    tokenizer, model, processor, _ = load_pretrained_model(
        'lmms-lab/LLaVA-Video-7B-Qwen2',
        None,
        'LLaVA-Video-7B-Qwen2',
        load_8bit=False, overwrite_config=overwrite_config, attn_implementation='sdpa')
    model.eval()
    #model = model.half()
    if not ARGS.cpu:
        model = model.cuda()

    tot_n_correct, tot = 0, 0
    all_scores = []
    os.makedirs(out_dir:=f'tvqa-results/llava', exist_ok=True)
    showseaseps = get_showseaseps(ARGS.show_name, ARGS.season, ARGS.ep)

    for show_name, seas, ep in (pbar:=tqdm(showseaseps)):
        season_qs = full_dset_qs[show_name_dict[show_name]][f'season_{seas}']
        if f'episode_{ep}' not in season_qs.keys():
            continue
        if (show_name, seas, ep) == ('house', 4, 11):
            continue

        ep_qs = season_qs[f'episode_{ep}']
        cache_fp = os.path.join(out_dir, f'{show_name}_s{seas:01}e{ep:01}.json')

        if os.path.exists(cache_fp) and not ARGS.recompute:
            with open(cache_fp) as f:
                x = f.read().split()
            new_correct, new_tot = int(x[0]), int(x[1])
        else:
            new_correct, new_tot = answer_qs(show_name, seas, ep, model, processor, tokenizer, ep_qs)
            with open(cache_fp, 'w') as f:
                f.write(f'{new_correct} {new_tot}')

        tot_n_correct += new_correct
        tot += new_tot
        all_scores.append([show_name, seas, ep, new_correct, new_tot, new_correct/new_tot])
        pbar.set_description(f'{show_name}-s{seas}e{ep}, running avg: {tot_n_correct}/{tot}={tot_n_correct/tot}')

    df = pd.DataFrame(all_scores, columns=['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    print(df.drop('show', axis=1).mean(axis=0))
    df.to_csv(f'{out_dir}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv')
