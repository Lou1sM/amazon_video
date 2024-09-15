from copy import copy
#import whisperx
from tqdm import tqdm
import json
import os
from os.path import join
import argparse
from natsort import natsorted
import numpy as np
import torch
import imageio_ffmpeg
from scipy.optimize import linear_sum_assignment
from PIL import Image
from dl_utils.misc import check_dir
from deepface import DeepFace # stupidly, this needs to be imported before "from transformers import AutoProcessor"
from caption_each_scene import Captioner
from episode import get_char_names
from scene_detection import SceneSegmenter
from utils import get_all_testnames


FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

def segment_and_save(vidname_list):
    scene_segmenter = SceneSegmenter()
    for vn in vidname_list:
        new_pt, new_timepoints = scene_segmenter.scene_segment(vn, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split)
        check_dir(kf_dir:=f'data/ffmpeg-keyframes-by-scene/{vn}')
        np.save(f'{kf_dir}/keyframe-timepoints.npy', new_timepoints)
        np.save(f'{kf_dir}/scene-split-timepoints.npy', new_pt)
        check_dir(cur_scene_dir:=f'{kf_dir}/{vn}_scene0')
        for fn in os.listdir(cur_scene_dir):
            os.remove(join(cur_scene_dir, fn))
        next_scene_idx = 1
        # move keyframes for each scene to their own dir
        for i, kf in enumerate(natsorted(os.listdir(scene_segmenter.framesdir))):
            if i in scene_segmenter.kf_scene_split_points:
                check_dir(cur_scene_dir:=f'{kf_dir}/{vn}_scene{next_scene_idx}')
                for fn in os.listdir(cur_scene_dir):
                    #print('removing',join(cur_scene_dir, fn))
                    os.remove(join(cur_scene_dir, fn))
                next_scene_idx += 1
            if kf != 'frametimes.npy':
                os.symlink(os.path.abspath(f'{scene_segmenter.framesdir}/{kf}'), os.path.abspath(f'{cur_scene_dir}/{kf}'))
    torch.cuda.empty_cache()

def segment_audio_transcript(vidname, recompute):
    scene_idx =
    scenes = []
    pt = np.load(f'data/ffmpeg-keyframes-by-scene/{vidname}/scene-split-timepoints.npy')
    pt = np.append(pt, np.inf)
    if os.path.exists(transc_no_names_fp:=f'data/transcripts/{vidname}-no-names.json') and not recompute:
        return
    with open(f'data/whisper_outputs/{vidname}.json') as f:
        audio_transcript = json.load(f)

    audio_transcript = [line for line in audio_transcript if not all(ord(x)>255 for x in line['text'].strip())]
    cur_scene = audio_transcript[:1]
    for i,ut in enumerate(audio_transcript[1:]):
        avg_time = (float(ut['start']) + float(ut['end'])) / 2
        if avg_time < pt[scene_idx]:
            cur_scene.append(ut)
        else:
            scenes.append(copy(cur_scene))
            scene_idx += 1
            cur_scene = [ut]
    scenes.append(cur_scene)
    transcript = '\n[SCENE_BREAK]\n'.join('\n'.join(x.get('speaker', 'UNK') + ': ' + x['text'] for x in s) for s in scenes).split('\n')
    if ARGS.print_transcript:
        print(transcript)

    tdata = {'Show Title': vidname, 'Transcript': transcript}
    with open(transc_no_names_fp, 'w') as f:
        json.dump(tdata, f)

def torch_load_jpg(fp):
    return torch.tensor(np.array(Image.open(fp)).transpose(2,0,1)).cuda().float()

def assign_char_names(vidname, recompute):
    if os.path.exists(assigned_transcr_fp:=f'data/transcripts/{vidname}.json') and not recompute:
        print(f'{assigned_transcr_fp} already exists')
        return
    with open(f'data/transcripts/{vidname}-no-names.json') as f:
        transcript = json.load(f)['Transcript']

    transcript_scenes = '£'.join(transcript).split('[SCENE_BREAK]')
    n_scenes = transcript.count('[SCENE_BREAK]') + 1
    check_dir(cfaces_dir:=f'data/scraped_char_faces/{vidname}')
    cface_fnames = natsorted(os.listdir(cfaces_dir))
    speakers_per_scene = [get_char_names(scene.split('£')) for scene in transcript_scenes]
    speakers_per_scene = [x for x in speakers_per_scene if x !=-1]
    assert len(speakers_per_scene) == n_scenes
    all_speakers = natsorted(list(set(c for scene in speakers_per_scene for c in scene if c!='UNK')))
    if cface_fnames == []:
        speaker_char_costs = speaker2char = {k:k for k in all_speakers}
    else:
        cface_feat_vecs = []
        names_to_remove = []
        for acn in cface_fnames:
            im_fps = [join(cfaces_dir, acn, x) for x in os.listdir(join(cfaces_dir, acn))]
            im_fps = [x for x in im_fps if x.endswith('.npy')]
            if len(im_fps) == 0:
                names_to_remove.append(acn)
                continue
            feat_vecs = np.array([ DeepFace.represent(np.load(fp), detector_backend='fastmtcnn')[0]['embedding'] for fp in im_fps])
            cface_feat_vecs.append(feat_vecs.mean(axis=0))
        cface_fnames = [x for x in cface_fnames if x not in names_to_remove]
        cface_feat_vecs = np.stack(cface_feat_vecs)
        char_scene_costs_list = []
        if os.path.exists(char_scene_costs_fp:=f'data/char-scene-costs/{vn}.npy'):
            print('loading from', char_scene_costs_fp)
            char_scene_costs = np.load(char_scene_costs_fp)
        else:
            print('extracting face features from scenes')
            for scene_idx in tqdm(range(n_scenes)):
                dfaces_dir = f'data/ffmpeg-keyframes-by-scene/{vidname}/{vidname}_scene{scene_idx}'
                dface_feat_vecs_list = []
                for dface_fn in natsorted(os.listdir(dfaces_dir)):
                    try:
                        dfaces = DeepFace.represent(img_path=join(dfaces_dir, dface_fn), detector_backend='fastmtcnn')
                    except ValueError: # means no face detected in the frame
                        continue
                    dface_feat_vecs_list += [x['embedding'] for x in dfaces]
                if len(dface_feat_vecs_list)==0: # think I can get away with not aligning
                    char_scene_costs_list.append(np.zeros(len(cface_fnames)))
                else:
                    dface_feat_vecs = np.array(dface_feat_vecs_list)
                    dists = np.expand_dims(dface_feat_vecs,0) - np.expand_dims(cface_feat_vecs,1)
                    dists = np.linalg.norm(dists, axis=2)
                    char_scene_costs_list.append(dists.min(axis=1))

            char_scene_costs = np.stack(char_scene_costs_list, axis=1)
            np.save(char_scene_costs_fp, char_scene_costs)
        speaker_char_costs_scenes = np.empty([len(all_speakers), len(cface_fnames)])
        for i,sid in enumerate(all_speakers):
            appearing_sidxs = [i for i, anames in enumerate(speakers_per_scene) if sid in anames]
            for j,act in enumerate(cface_fnames):
                cost = char_scene_costs[j, appearing_sidxs].mean()
                speaker_char_costs_scenes[i,j] = cost # of assigning speaker num i to char num j
        speaker_prominences = {s:'\n'.join(transcript).count(s) for s in all_speakers}
        speaker_prominences = {s:(c+1)/(sum(speaker_prominences.values())+1) for s,c in speaker_prominences.items()}
        with open(f'data/nimages-per-char/{vidname}-nimages-per-char.json') as f:
            nimages_per_char = json.load(f)
        assert all(x in nimages_per_char.keys() for x in set(cface_fnames))
        char_prominences = {k:(v+1)/(sum(nimages_per_char.values())+1) for k,v in nimages_per_char.items()}
        prominence_divergences = np.empty([len(all_speakers), len(cface_fnames)])
        for i,sid in enumerate(all_speakers):
            for j,act in enumerate(cface_fnames):
                prominence_divergences[i,j] = max(0, speaker_prominences[sid] - char_prominences[act])
        speaker_char_costs = speaker_char_costs_scenes + prominence_divergences
        nr = max(1,min(3,speaker_char_costs.shape[0]//speaker_char_costs.shape[1])) # max allowed number of speakers assigned to one char
        cost_mat = np.tile(speaker_char_costs, (1,nr))
        n_unassign = max(0, cost_mat.shape[0] - cost_mat.shape[1])
        no_assign_cost = np.tile(cost_mat.mean(axis=1, keepdims=True), (1,n_unassign))
        cost_mat = np.concatenate([cost_mat, no_assign_cost], axis=1)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        speaker2char = {}
        for ri in row_ind:
            speaker = all_speakers[ri]
            ci = col_ind[ri]
            if ci >= len(cface_fnames)*nr:
                char = 'UNASSIGNED'
                speaker2char[speaker] = speaker
            else:
                char_ind = ci % len(cface_fnames)
                char = cface_fnames[char_ind].removesuffix('.jpg')
                speaker2char[speaker] = char

    check_dir('data/speaker_char_mappings')
    with open(f'data/speaker_char_mappings/{vidname}.json', 'w') as f:
        json.dump(speaker2char, f)

    for k,v in speaker2char.items():
        transcript = [line.replace(k, v) for line in transcript]
    with_names_tdata = {'Show Title': vidname, 'Transcript': transcript}
    #with open('sol-autotranscript.txt') as f: gt = f.readlines()
    #gtn = [x.split(':')[0] for x in gt]
    #predn = [x.split(':')[0] for x in transcript]
    #n_correct = 0
    #denom = 0
    #for p,g in zip(predn, gtn):
    #    if p==g: n_correct+=1
    #    if 'SCENE_BREAK' not in p and p!='UNASSIGNED':
    #        denom += 1
    #print(f'n-lines: {len(gt)} n-predicted: {denom} n-correct: {n_correct} acc:, {n_correct/denom:.3f} harsh_acc: {n_correct/len(gtn):.3f}')
    with open(assigned_transcr_fp, 'w') as f:
        json.dump(with_names_tdata, f)

def caption_keyframes_and_save(vidname_list, recompute):
    captioner = Captioner()
    if recompute or not all(os.path.exists(f'data/video_scenes/{vn}/kosmos_procced_scene_caps.json') for vn in vidname_list):
        captioner.init_models('kosmos')

        for vn in vidname_list:
            raw_caps_fp = f'data/video_scenes/{vn}/kosmos_raw_scene_caps.json'
            if not os.path.exists(raw_caps_fp):
                captioner.kosmos_scene_caps(vn)
            procced_caps_fp = f'data/video_scenes/{vn}/kosmos_procced_scene_caps.json'
            if not os.path.exists(procced_caps_fp):
                #captioner.filter_and_namify_scene_captions(vn+'-auto-with-names', 'kosmos')
                captioner.filter_and_namify_scene_captions(vn, 'kosmos')
                if os.path.exists(f'data/transcripts/{vn}.json'):
                    captioner.filter_and_namify_scene_captions(vn, 'kosmos')

parser = argparse.ArgumentParser()
parser.add_argument('--recompute-keyframes', action='store_true')
parser.add_argument('--recompute-frame-features', action='store_true')
parser.add_argument('--recompute-best-split', action='store_true')
parser.add_argument('--recompute-captions', action='store_true')
parser.add_argument('--recompute-whisper', action='store_true')
parser.add_argument('--recompute-scene-summs', action='store_true')
parser.add_argument('--recompute-face-extraction', action='store_true')
parser.add_argument('--recompute-char-names', action='store_true')
parser.add_argument('--recompute-segment-trans', action='store_true')
parser.add_argument('--print-transcript', action='store_true')
parser.add_argument('--vidname', type=str, default='oltl-10-18-10')
parser.add_argument('--dbs', type=int, default=8)
parser.add_argument('--n-dpoints', type=int, default=3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('-t','--is_test', action='store_true')
parser.add_argument('--vid-name', type=str, default='the-silence-of-the-lambs_1991')
parser.add_argument('--llm-name', type=str, default='llama3-tiny', choices=['llama3-tiny', 'llama3-8b', 'llama3-70b'])
parser.add_argument('--summ-device', type=str, default='cuda', choices=['cuda', 'cpu'])
ARGS = parser.parse_args()

available_with_vids = [x.removesuffix('.mp4') for x in os.listdir('data/full_videos/')]
if ARGS.vid_name == 'all':
    test_vidnames, _ = get_all_testnames()
else:
    test_vidnames = [ARGS.vid_name]

segment_and_save(test_vidnames)
torch.cuda.empty_cache()
torch.cuda.empty_cache()
for vn in test_vidnames:
    segment_audio_transcript(vn, ARGS.recompute_segment_trans)
    #get_scene_faces(vn, ARGS.recompute_face_extraction)
    try:
        assign_char_names(vn, ARGS.recompute_char_names)
    except ValueError as e:
        print(e)

torch.cuda.empty_cache()
caption_keyframes_and_save(test_vidnames, ARGS.recompute_captions)
torch.cuda.empty_cache()

