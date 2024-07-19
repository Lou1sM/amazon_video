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
import whisperx
import face_recognition
import imageio_ffmpeg
#from deepface import DeepFace
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from dl_utils.misc import check_dir
from dl_utils.label_funcs import get_trans_dict_from_cost_mat, simple_get_trans_dict_from_cost_mat
from dl_utils.tensor_funcs import numpyify
from caption_each_scene import Captioner
from episode import episode_from_epname, get_char_names
from summarize_dialogue import SoapSummer
from scene_detection import SceneSegmenter
from utils import rouge_from_multiple_refs, display_rouges, prepare_for_pil



def segment_and_save(epname_list):
    scene_segmenter = SceneSegmenter()
    for en in epname_list:
        new_pt, new_timepoints = scene_segmenter.scene_segment(en, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split)
        check_dir(kf_dir:=f'data/keyframes/{en}')
        np.save(f'{kf_dir}/keyframe-timepoints.npy', new_timepoints)
        np.save(f'{kf_dir}/scene-split-timepoints.npy', new_pt)
        check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene0')
        next_scene_idx = 1
        print(f'found {len(scene_segmenter.kf_scene_split_points)+1} scenes')
        # move keyframes for each scene to their own dir
        for i, kf in enumerate(os.listdir(scene_segmenter.framesdir)):
            if i in scene_segmenter.kf_scene_split_points:
                check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene{next_scene_idx}')
                next_scene_idx += 1
            if kf != 'frametimes.npy':
                shutil.copy(f'{scene_segmenter.framesdir}/{kf}', cur_scene_dir)
    torch.cuda.empty_cache()

def whisper_and_save(epname_list, recompute):
    if all(os.path.exists(f'data/audio_transcripts/{en}.json') for en in epname_list):
        return
    model = whisperx.load_model("large-v2", 'cuda', compute_type='int8')
    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device='cuda')
    model_a, metadata = whisperx.load_align_model(language_code='en', device='cuda')

    for en in tqdm(epname_list):
        check_dir('data/audio_transcripts')
        audio_fpath = f'data/audio/{en}.wav'
        mp4_fpath = f'data/videos/{en}.mp4'

        if not os.path.exists(audio_fpath):
            FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
            extract_wav_cmd = f"{FFMPEG_PATH} -i {mp4_fpath} -ab 160k -ac 2 -ar 44100 -vn {audio_fpath}"
            subprocess.call(extract_wav_cmd, shell=True)
        if not recompute and os.path.exists(at_fpath:=f'data/audio_transcripts/{en}.json'):
            with open(at_fpath) as f:
                audio_transcript = json.load(f)
        else:
            audio = whisperx.load_audio(audio_fpath)

            result = model.transcribe(audio, batch_size=4)
            result = whisperx.align(result["segments"], model_a, metadata, audio, 'cuda', return_char_alignments=False)

            diarize_segments = diarize_model(audio)
            whisper_result = whisperx.assign_word_speakers(diarize_segments, result)['segments']

            cur_line = whisper_result[0]
            audio_transcript = []
            for ut in whisper_result[1:]:
                if ut.get('speaker', 'none1') == cur_line.get('speaker', 'none2'):
                    cur_line = {'start':cur_line['start'], 'end':ut['end'], 'text': cur_line['text'] + ' ' + ut['text'], 'speaker': cur_line['speaker']}
                else:
                    audio_transcript.append(cur_line)
                    cur_line = ut
            audio_transcript.append(cur_line)
            with open(at_fpath, 'w') as f:
                json.dump(audio_transcript, f)

def segment_audio_transcript(epname):
    scene_idx = 0
    scenes = []
    pt = np.load(f'data/keyframes/{epname}/scene-split-timepoints.npy')
    pt = np.append(pt, np.inf)
    with open(f'data/audio_transcripts/{epname}.json') as f:
        audio_transcript = json.load(f)

    cur_scene = audio_transcript[:1]
    for ut in audio_transcript[1:]:
        avg_time = (float(ut['start']) + float(ut['end'])) / 2
        if avg_time < pt[scene_idx+1]:
            cur_scene.append(ut)
        else:
            scenes.append(copy(cur_scene))
            scene_idx += 1
            cur_scene = [ut]
    scenes.append(cur_scene)
    transcript = '\n[SCENE_BREAK]\n'.join('\n'.join(x.get('speaker', 'UNK') + ': ' + x['text'] for x in s) for s in scenes).split('\n')
    if ARGS.print_transcript:
        print(transcript)

    #with open(f'data/transcripts/{epname}.json') as f:
        #tdata = json.load(f)
    tdata = {'Show Title': epname, 'Transcript': transcript}
    with open(f'data/transcripts/{epname}-auto.json', 'w') as f:
        json.dump(tdata, f)

def get_scene_faces(epname, recompute):
    keyframes_dir = f'data/keyframes/{epname}'
    faces_dir = f'data/faceframes/{epname}'
    kf_scene_dirs = natsorted([x for x in os.listdir(keyframes_dir) if '_scene' in x])
    mtcnn = MTCNN(image_size=160, margin=10, min_face_size=20, thresholds=[0.8, 0.8, 0.9], factor=0.709, post_process=True, device='cuda', keep_all=True)
    for kfsd in kf_scene_dirs:
        face_idx = 0
        scene_keyframes_dir = join(keyframes_dir, kfsd)
        scene_faces_dir = join(faces_dir, kfsd)
        if recompute and os.path.exists(scene_faces_dir):
            for fn in os.listdir(scene_faces_dir):
                os.remove(join(scene_faces_dir, fn))
        if (not os.path.exists(scene_faces_dir)) or recompute:
            check_dir(scene_faces_dir)
            batch_size = min(128, len(os.listdir(scene_keyframes_dir)))
            frames = []
            for keyframe_fn in os.listdir(scene_keyframes_dir):
                image = Image.open(join(scene_keyframes_dir, keyframe_fn))
                #import matplotlib.pyplot as plt
                #plt.imshow(image); plt.show()
                frames.append(image)
                if len(frames)%batch_size == 0:
                    #batched_face_locations = face_recognition.face_locations(frames[:3], model='cnn')
                    batched_faces = mtcnn(frames)
                    for frame_faces in batched_faces:
                        if frame_faces is None:
                            continue
                        for face in frame_faces: # all faces in frame are stack along first dim.
                            face = prepare_for_pil(face)
                            if face.shape[0] > 100 and face.shape[1] > 100:
                                save_fpath = join(scene_faces_dir, f'face{face_idx}.jpg')
                                print('saving face to', save_fpath, face.shape)
                                Image.fromarray(face).save(save_fpath)
                                face_idx += 1

def torch_load_jpg(fp):
    return torch.tensor(np.array(Image.open(fp)).transpose(2,0,1)).cuda().float()

def assign_char_names(epname, recompute):
    with open(f'data/transcripts/{epname}-auto.json') as f:
        transcript = json.load(f)['Transcript']

    transcript_scenes = '£'.join(transcript).split('[SCENE_BREAK]')
    speakers_per_scene = [get_char_names(scene.split('£')) for scene in transcript_scenes]
    speakers_per_scene = [x for x in speakers_per_scene if x !=-1]
    n_scenes = len(speakers_per_scene)
    assert n_scenes == transcript.count('[SCENE_BREAK]') + 1
    aface_fnames = natsorted(os.listdir(afaces_dir:=f'data/scraped_faces/{epname}'))
    actor_scene_costs_list = []
    face_resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    aface_ims = [torch_load_jpg(join(afaces_dir, aface_fn)) for aface_fn in aface_fnames]
    aface_feat_vecs = face_resnet(torch.stack(aface_ims))
    #for i, aface_fn in enumerate(aface_fnames):
    #mean_aface = aface_feat_vecs.mean(axis=0)
    #non_assign_scene_costs = []
    for scene_idx in range(n_scenes):
        dfaces_dir = f'data/faceframes/{epname}/{epname}_scene{scene_idx}'
        dface_ims = [torch_load_jpg(join(dfaces_dir, dface_fn)) for dface_fn in natsorted(os.listdir(dfaces_dir))]
        dface_feat_vecs = face_resnet(torch.stack(dface_ims))
        dists = dface_feat_vecs.unsqueeze(0) - aface_feat_vecs.unsqueeze(1)
        #dists = (dists**2).sum(axis=2)
        dists = np.linalg.norm(numpyify(dists), axis=2)
        #dists = torch.matmul(aface_feat_vecs, dface_feat_vecs.T)
        for i,j in enumerate(dists.argmin(axis=0)[:15]): print(i, aface_fnames[j])
        breakpoint()
        actor_scene_costs_list.append(numpyify(dists).min(axis=1))
        #nac = ((mean_aface - dface_feat_vecs)**2).sum(axis=1).min()
        #non_assign_scene_costs.append(dists.mean().item())
            #best_cost = 0
            #aface_fp = join(afaces_dir, aface_fn)
            #aface_im = torch_load_jpg(aface_fp)
            #aface_feat_vec = face_resnet(aface_im.unsqueeze(0))
            #for face_fn in os.listdir(detected_faces_dir):
            #    dface_path = join(detected_faces_dir, face_fn)
            #    dface_im = torch_load_jpg(dface_path)
            #    print(dface_path, dface_im.shape)
            #    #face_comparison_result = DeepFace.verify(img1_path=aface_fp, img2_path=dface_path)
            #    #new_cost = 1 - face_comparison_result['distance']
            #    dface_feat_vec = face_resnet(dface_im.unsqueeze(0))
            #    new_cost = ((aface_feat_vec - dface_feat_vec)**2).sum()
            #    if new_cost < best_cost:
            #        best_cost = new_cost
            #actor_scene_costs[i, scene_idx] = best_cost

    actor_scene_costs = np.stack(actor_scene_costs_list, axis=1)
    #non_assign_scene_costs = np.array(non_assign_scene_costs)
    all_speakers = natsorted(list(set(c for scene in speakers_per_scene for c in scene if c!=-1)))
    speaker_actor_costs = np.empty([len(all_speakers), len(aface_fnames)])
    #non_assign_speaker_costs = []
    for i,sid in enumerate(all_speakers):
        appearing_sidxs = [i for i, anames in enumerate(speakers_per_scene) if sid in anames]
        #nacs = non_assign_scene_costs[appearing_sidxs].sum()
        #non_assign_speaker_costs.append(nacs)
        for j,act in enumerate(aface_fnames):
            cost = actor_scene_costs[j, appearing_sidxs].mean()
            speaker_actor_costs[i,j] = cost # of assigning speaker num i to char num j
    #non_assign_speaker_costs = np.array(non_assign_speaker_costs)
    #n_non_assigned = speaker_actor_costs.shape[0] - speaker_actor_costs.shape[1]
    #tiled_non_assigned = np.tile(np.expand_dims(non_assign_speaker_costs,1),(1,n_non_assigned))
    nr = 3 # max allowed number of speakers assigned to one char
    tiled_sacs = np.tile(speaker_actor_costs, (1,nr))
    #cost_mat = np.concatenate([tiled_sacs, tiled_non_assigned], axis=1)
    cost_mat = tiled_sacs
    assert cost_mat.shape == (len(all_speakers), len(all_speakers)+(nr-1)*speaker_actor_costs.shape[1])
    trans_dict = get_trans_dict_from_cost_mat(cost_mat)
    #trans_dict = simple_get_trans_dict_from_cost_mat(speaker_actor_costs)
    speaker2char = {}
    for k,v in trans_dict.items():
        speaker = all_speakers[v]
        if k >= len(aface_fnames)*nr:
            char = 'UNASSIGNED'
        else:
            k = k%len(aface_fnames)
            char = aface_fnames[k].removesuffix('.jpg') # faces saved as {char-name}.jpg
        speaker2char[speaker] = char

    print(speaker2char)
    print(speaker2char['SPEAKER_17'])
    print(speaker2char['SPEAKER_39'])
    breakpoint()
    with open('data/speaker_char_mappings/{epname}.json', 'w') as f:
        json.dump(speaker2char, f)

    for k,v in speaker2char.items():
        transcript = transcript.replace(k, v)
    with_names_tdata = {'Show Title': epname, 'Transcript': transcript}
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
parser.add_argument('--print-transcript', action='store_true')
parser.add_argument('--epname', type=str, default='oltl-10-18-10')
parser.add_argument('--dbs', type=int, default=8)
parser.add_argument('--n-dpoints', type=int, default=3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('-t','--is_test', action='store_true')
parser.add_argument('--vid-name', type=str)
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
whisper_and_save(test_epnames, ARGS.recompute_whisper)
torch.cuda.empty_cache()
for en in test_epnames:
    segment_audio_transcript(en)
    get_scene_faces(en, ARGS.recompute_face_extraction)
    assign_char_names(en, ARGS.recompute_char_names)

torch.cuda.empty_cache()
caption_keyframes_and_save(test_epnames, ARGS.recompute_captions)
torch.cuda.empty_cache()


if ARGS.vid_name is None:
    ours, og = summarize_and_score(test_epnames)
    with open('results-only-video.txt', 'w') as f:
        for sname, sval in zip(['video only', 'with transcript'], [ours, og]):
            avg_rouge_scores = np.array(sval).mean(axis=0)
            to_display = ' '.join([f'{n}: {v:.4f}' for n,v in display_rouges(avg_rouge_scores)])
            to_display = f'{sname}:\n{to_display}\n'
            print(to_display)
            f.write(to_display + '\n')
else:
    high_level_summ_mod_name = 'lucadiliello/bart-small' if ARGS.is_test else 'facebook/bart-large-cnn'
    #high_level_summ_mod_name = 'lucadiliello/bart-small' if ARGS.is_test else 'chiakya/T5-large-Summarization'
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
                resumm_scenes=ARGS.recompute_scene_summs,
                do_save_new_scenes=True,
                is_test=ARGS.is_test)

    print(sum(x.numel() for x in summarizer_model.model.parameters()))
    concatted_scene_summs, final_summ = summarizer_model.summarize_from_epname(f'{en}-auto', min_len=500, max_len=600)
    print(concatted_scene_summs)
    with open(f'{ARGS.vid_name}-summary.txt', 'w') as f:
        f.write(final_summ)
    print(final_summ)

