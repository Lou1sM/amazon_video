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

parser = argparse.ArgumentParser()
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('--db_failed_scenes',action='store_true')
parser.add_argument('--print_full_aligned',action='store_true')
parser.add_argument('--ep_name',type=str, default='oltl-10-18-10')
ARGS = parser.parse_args()


img_res = 224
max_num_frames = 64


bert_model, config, tokenizer = get_bert_model(do_lower_case=True)
swin_model = get_swin_model(img_res, 'base', '600', False, True)
vl_transformer = VideoTransformer(True, config, swin_model, bert_model)
vl_transformer.freeze_backbone(freeze=False)
pretrained_model = torch.load('SwinBERT/models/table1/vatex/best-checkpoint/model.bin', map_location=torch.device('cpu'))
vl_transformer.load_state_dict(pretrained_model, strict=False)
vl_transformer.cuda()
vl_transformer.eval()

tensorizer = build_tensorizer(tokenizer, 50, 1568, 50, is_train=False)

scenes_dir = f'SummScreen/video_scenes/{ARGS.ep_name}'
caps_per_scene = []
ep = episode_from_ep_name(ARGS.ep_name)
#scene_fnames = sorted(os.listdir(scenes_dir),key=lambda x: int(x.split('_')[1].split('.')[0][5:]))
scene_fnames = [x for x in os.listdir(scenes_dir) if x.endswith('mp4')]
scene_nums = sorted([int(x.split('_')[1][5:-4]) for x in scene_fnames])
for sn in scene_nums:
    #scene_vid_path = os.path.join(scenes_dir,sfn)
    scene_vid_path = os.path.join(scenes_dir,f'{ARGS.ep_name}_scene{sn}.mp4')
    frames, _ = extract_frames_from_video_path(
                scene_vid_path, target_fps=3, num_frames=64,
                multi_thread_decode=False, sampling_strategy="uniform",
                safeguard_duration=False, start=None, end=None)
    cap = 'jim'
    scene_transcript = ep.scenes[sn]
    if sn==28:
        cap = inference(scene_vid_path, img_res, max_num_frames, vl_transformer, tokenizer, tensorizer)
        breakpoint()
    appearing_chars = set([x.split(':')[0] for x in scene_transcript.split('\n') if not x.startswith('[') and len(x) > 0])

    orig_cap = cap
    for cn in appearing_chars:
        g = gender(cn)
        if g=='m':
            cap = maybe_replace_subj(cn,'a man',cap)
            cap = maybe_replace_subj(cn,'a boy',cap)
        elif g=='f':
            cap = maybe_replace_subj(cn,'a woman',cap)
            cap = maybe_replace_subj(cn,'a girl',cap)
        elif len(appearing_chars)==1:
            cap = maybe_replace_subj(cn,'a person',cap)
            cap = maybe_replace_subj(cn,'someone',cap)
            cap = maybe_replace_subj(cn,'somebody',cap)
    print(f'SCENE{sn}: {cap}\t{orig_cap}')
    caps_per_scene.append(cap)

with open(f'{scenes_dir}/captions_per_scene.txt','w') as f:
    for i,c in enumerate(caps_per_scene):
        f.write(f'{i}: {c}')

