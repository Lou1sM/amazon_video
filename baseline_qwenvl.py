import os
from PIL import Image
from os.path import join
from tqdm import tqdm
import pandas as pd
import logging
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
import torch
import argparse
import json
from natsort import natsorted
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
#from qwen_vl_utils import process_vision_info


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

# Alternative to qwen_vl_utils functionality:
def process_vision_info(messages):
    image_inputs = [x["image"] for msg in messages for x in msg["content"] if x.get("type") == "image"]
    video_inputs = [x["video"] for msg in messages for x in msg["content"] if x.get("type") == "video"]
    return image_inputs, video_inputs

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

def answer_qs(show_name, season, episode, model, processor, ep_qs):
    dset_name = 'tvqa'
    vid_subpath = f'{dset_name}/{show_name}/season_{season}/episode_{episode}'

    # Get the visual and text information
    #vl_texts, _, scenes, viz_texts = get_texts('ours', vid_subpath)
    scenes = get_texts('ours', vid_subpath)
    scene_text = '[SCENE_BREAK]'.join('\n'.join(l for l in s) for s in scenes)
    scene_text = scene_text[-5000:]
    #viz_scene_text = '\n'.join(viz_texts)

    n_correct = 0

    # Get image paths (assuming they're in a directory)
    image_dir = f"data/ffmpeg-keyframes/{vid_subpath}"
    image_paths = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir)) if f.endswith('.jpg')]

    for i, qdict in enumerate(ep_qs['questions']):
        qsent = qdict['q']
        options = '\n'.join(f"{idx}: {qdict[f'a{idx}']}" for idx in range(5))

        # Prepare the multimodal input with images
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in image_paths[:1]],  # Use first 4 images
                    {"type": "text", "text": f"Context: {scene_text}"},
                    {"type": "text", "text": f"Question: {qsent}\nOptions:\n{options}\nAnswer with just a number (0-4)."}
                ]
            }
        ]

        # Process without video inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[Image.open(path) for path in image_paths[:4]],  # Load actual images
            return_tensors="pt",
        ).to(ARGS.device)

        # Inference
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Try to extract a number from the output
        try:
            ans = int(output_text.strip()[0])  # Get first character and try to convert to int
            if 0 <= ans <= 4:
                if ans == qdict['answer_idx']:
                    n_correct += 1
                if ARGS.verbose:
                    print(f"Question: {qsent}")
                    print(f"Options: {options}")
                    print(f"Model output: {output_text}")
                    print(f"Predicted: {ans}, Correct: {qdict['answer_idx']}\n")
            else:
                ans = -1  # Invalid answer
        except (ValueError, IndexError):
            ans = -1  # Couldn't parse answer

    n = len(ep_qs["questions"])
    print(f'VQA accuracy: {n_correct}/{n} = {n_correct/n:.5f}')
    return n_correct, n
        # Rest of your generation code...
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-name', type=str, default='friends')
    parser.add_argument('--season', type=int, default=2)
    parser.add_argument('--ep', type=int, default=-1)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--test-loading', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model', type=str, default='qwen-vl', choices=['qwen-vl', 'llama3-tiny', 'llama3-8b', 'llama3-70b'])
    parser.add_argument('--prec', type=int, default=4, choices=[32,8,4,2])
    parser.add_argument('--n-to-retrieve', type=int, default=1)
    parser.add_argument('--prompt-prefix', type=int, default=5000)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--dud', action='store_true')
    parser.add_argument("--rag-caches-prefix", type=str, default='.')
    parser.add_argument("--lava-outputs-prefix", type=str, default='.')
    ARGS = parser.parse_args()

    # Initialize the Qwen-VL model
    from transformers import BitsAndBytesConfig

    # Replace the model initialization part with this:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    if ARGS.cpu:
        model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            quantization_config=quantization_config,
            device_map="auto"
        )
    #model = Qwen2VLForConditionalGeneration.from_pretrained(
    #    "Qwen/Qwen2-VL-7B-Instruct",
    #    torch_dtype="auto",
    #    device_map="auto"
    #)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    ARGS.device = 'cpu' if ARGS.cpu else 'cuda'

    model.eval()

    tot_n_correct, tot = 0, 0
    all_scores = []

    os.makedirs(out_dir:=f'tvqa-results/{ARGS.model}', exist_ok=True)
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
            new_correct, new_tot = answer_qs(show_name, seas, ep, model, processor, ep_qs)
            with open(cache_fp, 'w') as f:
                f.write(f'{new_correct} {new_tot}')
        tot_n_correct += new_correct
        tot += new_tot
        all_scores.append([show_name, seas, ep, new_correct, new_tot, new_correct/new_tot])
        pbar.set_description(f'{show_name}-s{seas}e{ep}, running avg: {tot_n_correct}/{tot}={tot_n_correct/tot}')

    df = pd.DataFrame(all_scores, columns=['show', 'season', 'episode', 'n_correct', 'n', 'acc'])
    print(df.drop('show', axis=1).mean(axis=0))
    df.to_csv(f'{out_dir}/{ARGS.show_name}_{ARGS.season}-tvqa-results.csv')
