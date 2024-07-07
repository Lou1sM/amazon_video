from scene_detection import SceneSegmenter
from copy import copy
import json
import os
import whisperx
import subprocess
import imageio_ffmpeg
import numpy as np
import argparse
from dl_utils.misc import check_dir


parser = argparse.ArgumentParser()
parser.add_argument('--recompute-keyframes', action='store_true')
parser.add_argument('--recompute-frame-features', action='store_true')
parser.add_argument('--recompute-best-split', action='store_true')
parser.add_argument('--epname', type=str, default='oltl-10-18-10')
ARGS = parser.parse_args()

epname = 'oltl-10-18-10'
mp4_fpath = f'SummScreen/videos/{ARGS.epname}.mp4'
audio_fpath = f'SummScreen/audio/{ARGS.epname}.wav'
if not os.path.exists(audio_fpath):
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
    extract_wav_cmd = f"{FFMPEG_PATH} -i {mp4_fpath} -ab 160k -ac 2 -ar 44100 -vn {audio_fpath}"
    subprocess.call(extract_wav_cmd, shell=True)

ss = SceneSegmenter()
pt, timepoints = ss.scene_segment(ARGS.epname, recompute_keyframes=ARGS.recompute_keyframes, recompute_feats=ARGS.recompute_frame_features, recompute_best_split=ARGS.recompute_best_split)
check_dir(kf_dir:=f'SummScreen/video_scenes/{ARGS.epname}-auto/keyframes')
#next_end_of_scene = ss.kf_scene_split_points[0]
#check_dir('SummScreen/whisper_transcripts')
if os.path.exists(maybe_saved:=f'SummScreen/whisper_transcripts/{epname}.json'):
    with open(maybe_saved) as f:
        whisper_result = json.load(f)
else:
    device = "cuda"
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    audio = whisperx.load_audio(audio_fpath)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # add min/max number of speakers if known
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    result = model.transcribe(audio, batch_size=batch_size)
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)
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


pt = np.append(pt, np.inf)
scene_idx = 0
scenes = []
cur_scene = audio_transcript[:1]
for ut in audio_transcript[1:]:
    avg_time = (float(ut['start']) + float(ut['end'])) / 2
    if avg_time < pt[scene_idx+1]:
        cur_scene.append(ut)
        print(ut)
    else:
        scenes.append(copy(cur_scene))
        scene_idx += 1
        cur_scene = [ut]
scenes.append(cur_scene)
transcript = '\n[ SCENE_BREAK ]\n'.join('\n'.join(x.get('speaker', 'UNK') + ': ' + x['text'] for x in s) for s in scenes)
with open('SummScreen/automated_transcripts/{epname}.json', 'w') as f:
    f.write(transcript)
print(transcript)
breakpoint()

