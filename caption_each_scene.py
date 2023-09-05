import os
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

def caption_each_scene(ep_name, vl_transformer, tokenizer, tensorizer, img_res):
    max_num_frames = 64

    scenes_dir = f'SummScreen/video_scenes/{ep_name}'
    caps_per_scene = []
    ep = episode_from_ep_name(ep_name)
    #scene_fnames = sorted(os.listdir(scenes_dir),key=lambda x: int(x.split('_')[1].split('.')[0][5:]))
    scene_fnames = [x for x in os.listdir(scenes_dir) if x.endswith('mp4')]
    scene_nums = sorted([int(x.split('_')[1][5:-4]) for x in scene_fnames])
    if len(ep.scenes) == 1: # prob means no scene break markings
        print(f'There doesn\'t appear to be scene break markings for {ep_name}')
        return
    for sn in scene_nums:
        #scene_vid_path = os.path.join(scenes_dir,sfn)
        #if sn == 48:
        scene_vid_path = os.path.join(scenes_dir,f'{ep_name}_scene{sn}.mp4')
        frames, _ = extract_frames_from_video_path(
                    scene_vid_path, target_fps=3, num_frames=64,
                    multi_thread_decode=False, sampling_strategy="uniform",
                    safeguard_duration=False, start=None, end=None)
        cap = inference(scene_vid_path, img_res, max_num_frames, vl_transformer, tokenizer, tensorizer)
        #else:
            #cap = 'jim'
        scene_transcript = ep.scenes[sn]
        appearing_chars = set([x.split(':')[0] for x in scene_transcript.split('\n') if not x.startswith('[') and len(x) > 0])

        orig_cap = cap
        appearing_maybe_males = [c for c in appearing_chars if gender(c) in ['m','a']]
        appearing_maybe_females = [c for c in appearing_chars if gender(c) in ['f','a']]
        if len(appearing_maybe_males)==1:
            if appearing_maybe_males[0] in appearing_maybe_females:
                appearing_maybe_females.remove(appearing_maybe_males[0])
        if len(appearing_maybe_females)==1:
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
        #for cn in appearing_chars:
        #    g = gender(cn)
        #    if g=='m':
        #        cap = maybe_replace_subj(cn,'a man',cap)
        #        cap = maybe_replace_subj(cn,'a boy',cap)
        #    elif g=='f':
        #        cap = maybe_replace_subj(cn,'a woman',cap)
        #        cap = maybe_replace_subj(cn,'a girl',cap)
        #    elif len(appearing_chars)==1:
        #        cap = maybe_replace_subj(cn,'a person',cap)
        #        cap = maybe_replace_subj(cn,'someone',cap)
        #        cap = maybe_replace_subj(cn,'somebody',cap)
        print(f'SCENE{sn}: {cap}\t{orig_cap}')
        caps_per_scene.append(cap)

    with open(f'{scenes_dir}/captions_per_scene.txt','w') as f:
        for i,c in enumerate(caps_per_scene):
            f.write(f'{i}: {c}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--is_test',action='store_true')
    parser.add_argument('--db_failed_scenes',action='store_true')
    parser.add_argument('--print_full_aligned',action='store_true')
    parser.add_argument('--ep_name',type=str, default='oltl-10-18-10')
    ARGS = parser.parse_args()

    img_res = 224

    bert_model, config, tokenizer_ = get_bert_model(do_lower_case=True)
    swin_model = get_swin_model(img_res, 'base', '600', False, True)
    vl_transformer_ = VideoTransformer(True, config, swin_model, bert_model)
    vl_transformer_.freeze_backbone(freeze=False)
    pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cpu'))
    vl_transformer_.load_state_dict(pretrained_model, strict=False)
    vl_transformer_.cuda()
    vl_transformer_.eval()

    tensorizer_ = build_tensorizer(tokenizer_, 150, 1568, 50, is_train=False)

    if ARGS.ep_name == 'all':
        all_ep_names = [fn[:-4] for fn in os.listdir('SummScreen/video_scenes') if fn.endswith('.mp4')]
        for en in os.listdir('SummScreen/video_scenes'):
            caption_each_scene(en, vl_transformer_, tokenizer_, tensorizer_, img_res)
    else:
        caption_each_scene(ARGS.ep_name, vl_transformer_, tokenizer_, tensorizer_, img_res)

