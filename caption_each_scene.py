import os
from tqdm import tqdm
from time import time
from os.path import join
from copy import copy
import json
from nltk.corpus import names
from SwinBERT.src.tasks.run_caption_VidSwinBert_inference import inference
from SwinBERT.src.datasets.caption_tensorizer import build_tensorizer
from SwinBERT.src.modeling.load_swin import get_swin_model
from src.modeling.load_bert import get_bert_model
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
import torch
import argparse
from episode import episode_from_ep_name


male_names = names.words('male.txt')
female_names = names.words('female.txt')
def gender(char_name):
    if char_name in male_names and char_name not in female_names:
        return 'm'
    if char_name in female_names and char_name not in male_names:
        return 'f'
    else:
        return 'a'

def maybe_replace_subj(char_name, indeternp, cap):
    if cap.startswith(indeternp):
        return char_name + cap[len(indeternp):]
    else:
        return cap

def get_frames(vid_paths_list,n_frames):
    frames_list = []
    for vp in vid_paths_list:
        new_frames, _ = extract_frames_from_video_path(
                    vp, target_fps=3, num_frames=n_frames,
                    multi_thread_decode=True, sampling_strategy="uniform",
                    safeguard_duration=False, start=None, end=None)

        frames_list.append(new_frames)
    frames = torch.stack(frames_list)
    return frames

def caption_each_scene(ep_name, vl_transformer, tokenizer, tensorizer, img_res):
    #start_time = time()
    scenes_dir = f'SummScreen/video_scenes/{ep_name}'
    with open(f'SummScreen/transcripts/{ep_name}.json') as f:
        transcript_data = json.load(f)
    if not '[SCENE_BREAK]' in transcript_data['Transcript']:
        print(f'There doesn\'t appear to be scene break markings for {ep_name}')
        return
    scene_fnames = [x for x in os.listdir(scenes_dir) if x.endswith('mp4')]
    scene_nums = sorted([int(x.split('_')[1][5:-4]) for x in scene_fnames])
    scene_vid_paths = [os.path.join(scenes_dir,f'{ep_name}_scene{sn}.mp4') for sn in scene_nums]
    scene_caps = []
    #for i in range(0,len(scene_vid_paths),ARGS.bs):
        #batch_start_time = time()
        #batch_of_scene_vid_paths = scene_vid_paths[i:i+ARGS.bs]
        #frames_start_time = time()
    for vp in scene_vid_paths:
        frames, _ = extract_frames_from_video_path(
                    vp, target_fps=3, num_frames=n_frames,
                    multi_thread_decode=True, sampling_strategy="uniform",
                    safeguard_duration=False, start=None, end=None)
        if frames is None:
            newcap = ['']
            print(f'no scenes detected in {vp}, maybe it\'s v short')
        else:
            newcap = inference(frames, img_res, n_frames, vl_transformer, tokenizer, tensorizer)
        scene_caps += newcap


    #print(f'total time: {time()-start_time:.3f}')
    caps_per_scene = [{'scene_id': f'{ep_name}s{sn}', 'raw':c} for sn,c in enumerate(scene_caps)]
    with open(os.path.join(scenes_dir,'raw_captions_per_scene.json'), 'w') as f:
        json.dump(caps_per_scene,f)

def filter_and_namify_scene_captions(ep_name):
    scenes_dir = f'SummScreen/video_scenes/{ep_name}'
    ep = episode_from_ep_name(ep_name)
    with open(os.path.join(scenes_dir,'raw_captions_per_scene.json')) as f:
        raw_caps = [x['raw_cap'] for x in json.load(f)]
    assert len(raw_caps) == len(ep.scenes)
    caps_per_scene = []
    for sn, (raw_cap, scene_transcript) in enumerate(zip(raw_caps, ep.scenes)):
        appearing_chars = set([x.split(':')[0] for x in scene_transcript.split('\n') if not x.startswith('[') and len(x) > 0])

        cap = copy(raw_cap)
        appearing_maybe_males = [c for c in appearing_chars if gender(c) in ['m','a']]
        appearing_maybe_females = [c for c in appearing_chars if gender(c) in ['f','a']]
        if len(appearing_maybe_males)==1: # if has to be man then can't be woman
            if appearing_maybe_males[0] in appearing_maybe_females:
                appearing_maybe_females.remove(appearing_maybe_males[0])
        if len(appearing_maybe_females)==1: # and vice-versa
            if appearing_maybe_females[0] in appearing_maybe_males:
                appearing_maybe_males.remove(appearing_maybe_females[0])

        if len(appearing_maybe_males)==1:
            if 'a man' in cap:
                cap = cap.replace('a man',appearing_maybe_males[0], 1)
            elif 'a boy' in cap:
                cap = cap.replace('a boy',appearing_maybe_males[0], 1)
        if len(appearing_maybe_females)==1:
            if 'a woman' in cap:
                cap = cap.replace('a woman',appearing_maybe_females[0], 1)
            elif 'a girl' in cap:
                cap = cap.replace('a girl',appearing_maybe_females[0], 1)
        print(f'SCENE{sn}: {raw_cap}\t{cap}')
        caps_per_scene.append({'scene_id': f'{ep_name}s{sn}', 'raw':raw_cap, 'with_names':cap})

    with open(f'{scenes_dir}/procced_captions_per_scene.json','w') as f:
        json.dump(caps_per_scene,f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--ep_name',type=str, default='oltl-10-18-10')
    parser.add_argument('--bs',type=int, default=1)
    ARGS = parser.parse_args()


    img_res = 224
    n_frames = 32
    img_seq_len = int((n_frames/2)*(int(img_res)/32)*(int(img_res)/32))
    max_gen_len = 50

    bert_model, config, tokenizer_ = get_bert_model(do_lower_case=True)
    swin_model = get_swin_model(img_res, 'base', '600', False, True)
    vl_transformer_ = VideoTransformer(True, config, swin_model, bert_model)
    vl_transformer_.freeze_backbone(freeze=False)
    pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cpu'))
    vl_transformer_.load_state_dict(pretrained_model, strict=False)
    vl_transformer_.cuda()
    vl_transformer_.eval()

    tensorizer_ = build_tensorizer(tokenizer_, 150, img_seq_len, max_gen_len, is_train=False)

    if ARGS.ep_name == 'all':
        all_ep_names = os.listdir('SummScreen/video_scenes')
        to_caption = []
        for en in all_ep_names:
            if os.path.exists(f'SummScreen/video_scenes/{en}/raw_captions_per_scene.json'):
                print(f'scene captions already exist for {en}')
            else:
                to_caption.append(en)
            
        for tc in tqdm(to_caption):
            caption_each_scene(tc, vl_transformer_, tokenizer_, tensorizer_, img_res)
    else:
        caption_each_scene(ARGS.ep_name, vl_transformer_, tokenizer_, tensorizer_, img_res)

