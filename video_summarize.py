from copy import copy
from tqdm import tqdm
import json
import os
from os.path import join
import shutil
import subprocess
import argparse
from natsort import natsorted
import numpy as np
import pandas as pd
import torch
import imageio_ffmpeg
from scipy.optimize import linear_sum_assignment
from faces_train.train import FaceLearner, read_tfloat_im
from PIL import Image
from dl_utils.misc import check_dir
from dl_utils.label_funcs import get_trans_dict_from_cost_mat, simple_get_trans_dict_from_cost_mat
from dl_utils.tensor_funcs import numpyify, display_image
from deepface import DeepFace # stupidly, this needs to be imported before "from transformers import AutoProcessor"
from caption_each_scene import Captioner
from episode import episode_from_epname, get_char_names
from summarize_dialogue import SoapSummer
from scene_detection import SceneSegmenter
from utils import rouge_from_multiple_refs, display_rouges, bernoulli_CE


FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

#mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, thresholds=[0.8, 0.8, 0.9], factor=0.709, post_process=True, device='cuda', keep_all=True)

def segment_and_save(epname_list):
    scene_segmenter = SceneSegmenter()
    for en in epname_list:
        new_pt, new_timepoints = scene_segmenter.scene_segment(en, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split)
        check_dir(kf_dir:=f'data/keyframes-by-scene/{en}')
        np.save(f'{kf_dir}/keyframe-timepoints.npy', new_timepoints)
        np.save(f'{kf_dir}/scene-split-timepoints.npy', new_pt)
        check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene0')
        next_scene_idx = 1
        print(f'found {len(scene_segmenter.kf_scene_split_points)+1} scenes')
        # move keyframes for each scene to their own dir
        for i, kf in enumerate(natsorted(os.listdir(scene_segmenter.framesdir))):
            if i in scene_segmenter.kf_scene_split_points:
                check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene{next_scene_idx}')
                next_scene_idx += 1
            if kf != 'frametimes.npy':
                shutil.copy(f'{scene_segmenter.framesdir}/{kf}', cur_scene_dir)
    torch.cuda.empty_cache()

def segment_audio_transcript(epname, recompute):
    scene_idx = 0
    scenes = []
    pt = np.load(f'data/keyframes-by-scene/{epname}/scene-split-timepoints.npy')
    pt = np.append(pt, np.inf)
    vid_fpath = f"data/videos/{epname}.mp4"
    check_dir(utframe_dir:=f'data/utterance-frames/{epname}')
    if os.path.exists(utframe_dir) and not recompute:
        return
    with open(f'data/audio_transcripts/{epname}.json') as f:
        audio_transcript = json.load(f)

    audio_transcript = [line for line in audio_transcript if not all(ord(x)>255 for x in line['text'].strip())]
    cur_scene = audio_transcript[:1]
    for i,ut in enumerate(audio_transcript[1:]):
        avg_time = (float(ut['start']) + float(ut['end'])) / 2
        fp = f'data/utterance-frames/{epname}/ut{i}.jpg'
        #ffmpeg -ss 01:23:45 -i input -frames:v 1 -q:v 2 output.jpg
        #subprocess.run([FFMPEG_PATH, '-ss', str(avg_time), '-i', vid_fpath, '-frames:v', '1', '-q:v', '2', fp, '-y'])
        if avg_time < pt[scene_idx]:
            cur_scene.append(ut)
        else:
            scenes.append(copy(cur_scene))
            scene_idx += 1
            cur_scene = [ut]
    breakpoint()
    scenes.append(cur_scene)
    transcript = '\n[SCENE_BREAK]\n'.join('\n'.join(x.get('speaker', 'UNK') + ': ' + x['text'] for x in s) for s in scenes).split('\n')
    if ARGS.print_transcript:
        print(transcript)

    tdata = {'Show Title': epname, 'Transcript': transcript}
    with open(f'data/transcripts/{epname}-auto.json', 'w') as f:
        json.dump(tdata, f)

def get_scene_faces(epname, recompute):
    keyframes_dir = f'data/keyframes-by-scene/{epname}'
    faces_dir = f'data/faceframes/{epname}'
    kf_scene_dirs = natsorted([x for x in os.listdir(keyframes_dir) if '_scene' in x])
    for kfsd in kf_scene_dirs:
        face_idx = 0
        scene_keyframes_dir = join(keyframes_dir, kfsd)
        scene_faces_dir = join(faces_dir, kfsd)
        if recompute and os.path.exists(scene_faces_dir):
            for fn in os.listdir(scene_faces_dir):
                os.remove(join(scene_faces_dir, fn))
        if (not os.path.exists(scene_faces_dir)) or recompute:
            check_dir(scene_faces_dir)
            #batch_size = min(128, len(os.listdir(scene_keyframes_dir)))
            #frames = []
            for keyframe_fn in os.listdir(scene_keyframes_dir):
                #image = Image.open(join(scene_keyframes_dir, keyframe_fn))
                #import matplotlib.pyplot as plt
                #plt.imshow(image); plt.show()
                #frames.append(image)
                #if len(frames)%batch_size == 0:
                    #batched_face_locations = face_recognition.face_locations(frames[:3], model='cnn')
                    #batched_faces = mtcnn(frames)
                fpath = join(scene_keyframes_dir, keyframe_fn)
                frame_faces = DeepFace.extract_faces(fpath, detector_backend='fastmtcnn', enforce_detection=False)
                for dface in frame_faces:
                    if dface['confidence'] < 0.5:
                        continue
                    face_img = dface['face']
                    breakpoint()
                    pil_face_img = Image.fromarray((face_img*255).astype(np.uint8) )
                    if face_img.shape[0] > 100 and face_img.shape[1] > 100:
                        save_fpath = join(scene_faces_dir, f'face{face_idx}.jpg')
                        print('saving face to', save_fpath, face_img.shape)
                        #Image.fromarray(face).save(save_fpath)
                        pil_face_img.save(save_fpath)
                        face_idx += 1

def torch_load_jpg(fp):
    return torch.tensor(np.array(Image.open(fp)).transpose(2,0,1)).cuda().float()

def assign_char_names(epname):
    with open(f'data/transcripts/{epname}-auto.json') as f:
        transcript = json.load(f)['Transcript']

    transcript_scenes = '£'.join(transcript).split('[SCENE_BREAK]')
    n_scenes = transcript.count('[SCENE_BREAK]') + 1
    check_dir(cfaces_dir:=f'data/scraped_char_faces/{epname}')
    cface_fnames = natsorted(os.listdir(cfaces_dir))
    cface_feat_vecs = []
    names_to_remove = []
    for acn in cface_fnames:
        im_fps = [join(cfaces_dir, acn, x) for x in os.listdir(join(cfaces_dir, acn))]
        im_fps = [x for x in im_fps if x.endswith('.npy')]
        if len(im_fps) == 0:
            names_to_remove.append(acn)
            continue
        feat_vecs = np.array([ DeepFace.represent(np.load(fp))[0]['embedding'] for fp in im_fps])
        cface_feat_vecs.append(feat_vecs.mean(axis=0))
    cface_fnames = [x for x in cface_fnames if x not in names_to_remove]
    cface_feat_vecs = np.stack(cface_feat_vecs)
    char_scene_costs_list = []
    for scene_idx in range(n_scenes):
        dfaces_dir = f'data/keyframes-by-scene/{epname}/{epname}_scene{scene_idx}'
        dface_feat_vecs_list = []
        for dface_fn in natsorted(os.listdir(dfaces_dir)):
            try:
                dfaces = DeepFace.represent(img_path=join(dfaces_dir, dface_fn))
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
    with open(f'data/transcripts/{epname}-auto.json') as f:
        transcript = json.load(f)['Transcript']
    speakers_per_scene = [get_char_names(scene.split('£')) for scene in transcript_scenes]
    speakers_per_scene = [x for x in speakers_per_scene if x !=-1]
    assert len(speakers_per_scene) == n_scenes
    all_speakers = natsorted(list(set(c for scene in speakers_per_scene for c in scene if c!='UNK')))
    speaker_char_costs_scenes = np.empty([len(all_speakers), len(cface_fnames)])
    for i,sid in enumerate(all_speakers):
        appearing_sidxs = [i for i, anames in enumerate(speakers_per_scene) if sid in anames]
        for j,act in enumerate(cface_fnames):
            cost = char_scene_costs[j, appearing_sidxs].mean()
            speaker_char_costs_scenes[i,j] = cost # of assigning speaker num i to char num j
    speaker_prominences = {s:'\n'.join(transcript).count(s) for s in all_speakers}
    speaker_prominences = {s:(c+1)/(sum(speaker_prominences.values())+1) for s,c in speaker_prominences.items()}
    with open(f'data/nimages-per-char/{epname}-nimages-per-char.json') as f:
        nimages_per_char = json.load(f)
    assert all(x in nimages_per_char.keys() for x in set(cface_fnames))
    char_prominences = {k:(v+1)/(sum(nimages_per_char.values())+1) for k,v in nimages_per_char.items()}
    prominence_divergences = np.empty([len(all_speakers), len(cface_fnames)])
    for i,sid in enumerate(all_speakers):
        for j,act in enumerate(cface_fnames):
            #prominence_divergences[i,j] = bernoulli_CE(speaker_prominences[sid], char_prominences[act])
            #prominence_divergences[i,j] = abs(speaker_prominences[sid] - char_prominences[act])
            prominence_divergences[i,j] = max(0, speaker_prominences[sid] - char_prominences[act])
    nr = 3 # max allowed number of speakers assigned to one char
    #speaker_char_costs = speaker_char_costs_shots + speaker_char_costs_scenes*0.1
    speaker_char_costs = speaker_char_costs_scenes + prominence_divergences
    #speaker_char_costs = prominence_divergences
    cost_mat = np.tile(speaker_char_costs, (1,nr))
    n_unassign = cost_mat.shape[0] - cost_mat.shape[1]
    no_assign_cost = np.tile(cost_mat.mean(axis=1, keepdims=True), (1,n_unassign))
    cost_mat = np.concatenate([cost_mat, no_assign_cost], axis=1)
    #assert cost_mat.shape == (len(all_speakers), len(all_speakers)+(nr-1)*speaker_actor_costs.shape[1])
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    #trans_dict = simple_get_trans_dict_from_cost_mat(cost_mat)
    speaker2char = {}
    for ri in row_ind:
        speaker = all_speakers[ri]
        ci = col_ind[ri]
        if ci >= len(cface_fnames)*nr:
            char = 'UNASSIGNED'
        else:
            char_ind = ci % len(cface_fnames)
            char = cface_fnames[char_ind].removesuffix('.jpg')
        speaker2char[speaker] = char

    print(speaker2char)
    x = speaker_char_costs
    breakpoint()
    print(speaker2char['SPEAKER_17'])
    print(speaker2char['SPEAKER_39'])
    check_dir('data/speaker_char_mappings')
    with open('data/speaker_char_mappings/{epname}.json', 'w') as f:
        json.dump(speaker2char, f)

    for k,v in speaker2char.items():
        transcript = [line.replace(k, v) for line in transcript]
    with_names_tdata = {'Show Title': epname, 'Transcript': transcript}
    breakpoint()
    with open('sol-autotranscript.txt') as f: gt = f.readlines()
    gtn = [x.split(':')[0] for x in gt]
    predn = [x.split(':')[0] for x in transcript]
    n_correct = 0
    denom = 0
    for p,g in zip(predn, gtn):
        if p==g: n_correct+=1
        if 'SCENE_BREAK' not in p and p!='UNASSIGNED':
            denom += 1
    print(f'n-lines: {len(gt)} n-predicted: {denom} n-correct: {n_correct} acc:, {n_correct/denom:.3f} harsh_acc: {n_correct/len(gtn):.3f}')
    with open(f'data/transcripts/{epname}-auto-with-names.json', 'w') as f:
        json.dump(with_names_tdata, f)

def caption_keyframes_and_save(epname_list, recompute):
    captioner = Captioner()
    if recompute or not all(os.path.exists(f'data/video_scenes/{en}-auto/kosmos_procced_scene_caps.json') for en in epname_list):
        captioner.init_models('kosmos')

        for en in epname_list:
            raw_caps_fp = f'data/video_scenes/{en}/kosmos_raw_scene_caps.json'
            if not os.path.exists(raw_caps_fp):
                captioner.kosmos_scene_caps(en)
            procced_caps_fp = f'data/video_scenes/{en}/kosmos_procced_scene_caps.json'
            if not os.path.exists(procced_caps_fp):
                captioner.filter_and_namify_scene_captions(en+'-auto-with-names', 'kosmos')
                if os.path.exists(f'data/transcripts/{en}.json'):
                    captioner.filter_and_namify_scene_captions(en, 'kosmos')
            check_dir(f'data/video_scenes/{en}-auto')
            shutil.copy(procced_caps_fp, f'data/video_scenes/{en}-auto/kosmos_procced_scene_caps.json')

def summarize_and_score(epname_list):
    high_level_summ_mod_name = 'lucadiliello/bart-small' if ARGS.is_test else 'facebook/bart-large-cnn'
    summarizer_model = SoapSummer(model_name=high_level_summ_mod_name,
                    device='cuda',
                    bs=ARGS.bs,
                    dbs=ARGS.dbs,
                    #tokenizer=tokenizer,
                    caps='kosmos',
                    scene_order='identity',
                    uniform_breaks=False,
                    startendscenes=False,
                    centralscenes=False,
                    max_chunk_size=10000,
                    expdir='single-ep',
                    data_dir='./data',
                    resumm_scenes=False,
                    do_save_new_scenes=True,
                    is_test=ARGS.is_test)

    pbar = tqdm(epname_list)
    all_our_scores = []
    all_og_scores = []
    for en in pbar:
        ep = episode_from_epname(f'{en}-auto', infer_splits=False)
        concatted_scene_summs, final_summ = summarizer_model.summarize_from_epname(f'{en}-auto', min_len=360, max_len=400)
        concatted_scene_summs, og_final_summ = summarizer_model.summarize_from_epname(f'{en}', min_len=360, max_len=400)
        new_our_scores = rouge_from_multiple_refs(final_summ, ep.summaries.values(), benchmark_rl=False, return_full=False)
        new_og_scores = rouge_from_multiple_refs(og_final_summ, ep.summaries.values(), benchmark_rl=False, return_full=False)
        all_our_scores.append(new_our_scores)
        all_og_scores.append(new_og_scores)
        pbar.set_description(f'Ours: {new_our_scores[1]:.4f} OG: {new_og_scores[1]:.4f}')
    return all_our_scores, all_og_scores

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
parser.add_argument('--epname', type=str, default='oltl-10-18-10')
parser.add_argument('--dbs', type=int, default=8)
parser.add_argument('--n-dpoints', type=int, default=3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('-t','--is_test', action='store_true')
parser.add_argument('--vid-name', type=str, default='silence-of-lambs')
ARGS = parser.parse_args()

available_with_vids = [x.removesuffix('.mp4') for x in os.listdir('data/videos/')]
if ARGS.vid_name is None:
    dset_info = pd.read_csv('dset_info.csv', index_col=0)
    #test_mask = dset_info['usable'] & (dset_info['split']=='test')
    test_mask = dset_info['usable']
    test_epnames = dset_info.index[test_mask]

    test_epnames = [x for x in test_epnames if x in available_with_vids]
    test_epnames = test_epnames[:ARGS.n_dpoints]
else:
    test_epnames = [ARGS.vid_name]

segment_and_save(test_epnames)
torch.cuda.empty_cache()
#whisper_and_save(test_epnames, ARGS.recompute_whisper)
torch.cuda.empty_cache()
for en in test_epnames:
    segment_audio_transcript(en, ARGS.recompute_segment_trans)
    get_scene_faces(en, ARGS.recompute_face_extraction)
    if ARGS.recompute_char_names:
        assign_char_names(en)

torch.cuda.empty_cache()
caption_keyframes_and_save(test_epnames, ARGS.recompute_captions)
torch.cuda.empty_cache()


#if ARGS.vid_name is None:
#    ours, og = summarize_and_score(test_epnames)
#    with open('results-only-video.txt', 'w') as f:
#        for sname, sval in zip(['video only', 'with transcript'], [ours, og]):
#            avg_rouge_scores = np.array(sval).mean(axis=0)
#            to_display = ' '.join([f'{n}: {v:.4f}' for n,v in display_rouges(avg_rouge_scores)])
#            to_display = f'{sname}:\n{to_display}\n'
#            print(to_display)
#            f.write(to_display + '\n')
#else:
summarizer_model = SoapSummer(
            device='cuda',
            bs=ARGS.bs,
            dbs=ARGS.dbs,
            #tokenizer=tokenizer,
            caps='kosmos',
            scene_order='identity',
            uniform_breaks=False,
            startendscenes=False,
            centralscenes=False,
            max_chunk_size=10000,
            expdir='single-ep',
            data_dir='./data',
            resumm_scenes=ARGS.recompute_scene_summs,
            do_save_new_scenes=True,
            is_test=ARGS.is_test)

print(sum(x.numel() for x in summarizer_model.model.parameters()))
concatted_scene_summs, final_summ = summarizer_model.summarize_from_epname(f'{en}-auto', min_len=500, max_len=600)
print(concatted_scene_summs)
with open(f'{ARGS.vid_name}-summary.txt', 'w') as f:
    f.write(final_summ)
print(final_summ)

