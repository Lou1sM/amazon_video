from scene_detection import SceneSegmenter
from copy import copy
from tqdm import tqdm
import json
import os
import shutil
import whisperx
import subprocess
import argparse
import numpy as np
import pandas as pd
import torch
import imageio_ffmpeg
from dl_utils.misc import check_dir
from caption_each_scene import Captioner
from episode import episode_from_epname
from summarize_dialogue import SoapSummer
from utils import rouge_from_multiple_refs, display_rouges



def segment_and_save(epname_list):
    scene_segmenter = SceneSegmenter()
    all_pts = []
    all_timepoints = []
    for en in epname_list:
        new_pt, new_timepoints = scene_segmenter.scene_segment(en, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split)
        all_pts.append(new_pt)
        all_timepoints.append(new_timepoints)
        check_dir(kf_dir:=f'data/keyframes/{en}')
        check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene0')
        next_scene_idx = 1
        print(f'found {len(scene_segmenter.kf_scene_split_points)+1} scenes')
        for i, kf in enumerate(os.listdir(scene_segmenter.framesdir)):
            if i in scene_segmenter.kf_scene_split_points:
                check_dir(cur_scene_dir:=f'{kf_dir}/{en}_scene{next_scene_idx}')
                next_scene_idx += 1
            shutil.copy(f'{scene_segmenter.framesdir}/{kf}', cur_scene_dir)
    torch.cuda.empty_cache()
    return all_pts, all_timepoints

def whisper_and_save(epname_list, recompute):
    # move keyframes for each scene to their own dir
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
    pt = np.load(f'data/inferred-vid-splits/{epname}-inferred-vid-splits.npy')
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

def caption_keyframes_and_save(epname_list, recompute):
    #epname_list = epname_list_ + [x+'-auto' for x in epname_list_]
    captioner = Captioner()
    if recompute or not all(os.path.exists(f'data/video_scenes/{en}-auto/kosmos_procced_scene_caps.json') for en in epname_list):
        captioner.init_models('kosmos')

        for en in epname_list:
            raw_caps_fp = f'data/video_scenes/{en}/kosmos_raw_scene_caps.json'
            if not os.path.exists(raw_caps_fp):
                captioner.kosmos_scene_caps(en)
            procced_caps_fp = f'data/video_scenes/{en}/kosmos_procced_scene_caps.json'
            if not os.path.exists(procced_caps_fp):
                captioner.filter_and_namify_scene_captions(en+'-auto', 'kosmos')
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
        concatted_scene_summs, final_summ = summarizer_model.summarize_from_epname(f'{en}-auto')
        concatted_scene_summs, og_final_summ = summarizer_model.summarize_from_epname(f'{en}')
        new_our_scores = rouge_from_multiple_refs(final_summ, ep.summaries.values(), benchmark_rl=False, return_full=False)
        new_og_scores = rouge_from_multiple_refs(og_final_summ, ep.summaries.values(), benchmark_rl=False, return_full=False)
        all_our_scores.append(new_our_scores)
        all_og_scores.append(new_og_scores)
        pbar.set_description(f'Ours: {new_our_scores[1]:.4f} OG: {new_og_scores[1]:.4f}')
    return all_our_scores, all_og_scores

#dset_info = pd.read_csv('dset_info.csv', index_col=0)
#test_mask = dset_info['usable'] & (dset_info['split']=='test')
#test_epnames = dset_info.index[test_mask]

parser = argparse.ArgumentParser()
parser.add_argument('--recompute-keyframes', action='store_true')
parser.add_argument('--recompute-frame-features', action='store_true')
parser.add_argument('--recompute-best-split', action='store_true')
parser.add_argument('--recompute-captions', action='store_true')
parser.add_argument('--recompute-whisper', action='store_true')
parser.add_argument('--recompute-scene-summs', action='store_true')
parser.add_argument('--print-transcript', action='store_true')
parser.add_argument('--epname', type=str, default='oltl-10-18-10')
parser.add_argument('--dbs', type=int, default=8)
parser.add_argument('--n-dpoints', type=int, default=3)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('-t','--is_test', action='store_true')
parser.add_argument('--vid-name', type=str, default=3)
ARGS = parser.parse_args()

test_epnames = [x.removesuffix('.mp4') for x in os.listdir('data/videos/')][:ARGS.n_dpoints] if ARGS.vid_name is None else [ARGS.vid_name]

segment_and_save(test_epnames)
torch.cuda.empty_cache()
whisper_and_save(test_epnames, ARGS.recompute_whisper)
torch.cuda.empty_cache()
for en in test_epnames:
    segment_audio_transcript(en)
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

    concatted_scene_summs, final_summ = summarizer_model.summarize_from_epname(f'{en}-auto')
    print(final_summ)

