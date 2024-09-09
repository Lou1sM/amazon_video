import os
import argparse
import sys
import subprocess
import json
import imageio_ffmpeg
import whisperx
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--vid-name', type=str, required=True)
#parser.add_argument('--vid-name-list-file', type=str)
parser.add_argument('--recompute', action='store_true')
parser.add_argument('--delete-wav-after', action='store_true')
parser.add_argument('--bs', type=int, default=16)
ARGS = parser.parse_args()

#if (ARGS.vid_name==None) == (ARGS.vid_name_list_file==None):
    #sys.exit("You must specify exactly one of vid-name or vid-name-list-file")
#if ARGS.vid_name is not None:
if ARGS.vid_name == 'all':
    with open('clean-vid-names-to-command-line-names.json') as f:
        official2cl = json.load(f)
    vid_names = official2cl.values()
else:
    vid_names = [ARGS.vid_name]

device = "cuda"
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
for vn in tqdm(vid_names):
    if os.path.exists(transc_fpath:=f'data/whisper_outputs/{vn}.json') and not ARGS.recompute:
        print(f'Transcription already exists at {transc_fpath}, skipping')
        continue
    audio_fpath = f'data/audio/{vn}.wav'

    if not os.path.exists(audio_fpath):
        mp4_fpath = f'data/full_videos/{vn}.mp4'
        print(f'No file found at {audio_fpath}\nExtracting audio from {mp4_fpath}')
        FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
        extract_wav_cmd = f"{FFMPEG_PATH} -i {mp4_fpath} -ab 160k -ac 2 -ar 44100 -vn {audio_fpath} -loglevel quiet"
        subprocess.call(extract_wav_cmd, shell=True)
    audio = whisperx.load_audio(audio_fpath)

    result = model.transcribe(audio, batch_size=ARGS.bs)
    #model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    #print(diarize_segments)
    segments_result = result["segments"]
    #print(result["segments"])
    cur_line = result['segments'][0]
    audio_transcript = []
    for ut in result['segments'][1:]:
        if ut.get('speaker', 'none1') == cur_line.get('speaker', 'none2'):
            cur_line = {'start':cur_line['start'], 'end':ut['end'], 'text': cur_line['text'] + ' ' + ut['text'], 'speaker': cur_line['speaker']}
        else:
            audio_transcript.append(cur_line)
            cur_line = ut
    audio_transcript.append(cur_line)
    with open(transc_fpath, 'w') as f:
        json.dump(audio_transcript, f)

    if ARGS.delete_wav_after:
        os.remove(audio_fpath)

breakpoint()

