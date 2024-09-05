#import subprocess
import argparse
#import os
import json
#import imageio_ffmpeg
import whisperx
#from dl_utils.misc import check_dir


#def whisper_and_save(epname, recompute):

parser = argparse.ArgumentParser()
parser.add_argument('--vid-name', type=str, default='silence-of-lambs')
parser.add_argument('--recompute', action='store_true')
ARGS = parser.parse_args()

model = whisperx.load_model("large-v2", 'cuda', compute_type='int8')
diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device='cuda')
model_a, metadata = whisperx.load_align_model(language_code='en', device='cuda')

#check_dir('data/audio_transcripts')
audio_fpath = f'data/audio/{ARGS.vid_name}.wav'
mp4_fpath = f'data/videos/{ARGS.vid_name}.mp4'

#if not os.path.exists(audio_fpath):
#    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
#    extract_wav_cmd = f"{FFMPEG_PATH} -i {mp4_fpath} -ab 160k -ac 2 -ar 44100 -vn {audio_fpath}"
#    subprocess.call(extract_wav_cmd, shell=True)
#if not ARGS.recompute and os.path.exists(at_fpath:=f'data/audio_transcripts/{ARGS.vid_name}.json'):
#    with open(at_fpath) as f:
#        audio_transcript = json.load(f)
#else:
audio = whisperx.load_audio(audio_fpath)

result = model.transcribe(audio, batch_size=4)
result = whisperx.align(result["segments"], model_a, metadata, audio, 'cuda', return_char_alignments=False)

diarize_segments = diarize_model(audio)
whisper_result = whisperx.assign_word_speakers(diarize_segments, result)['segments']

cur_line = whisper_result[0]
audio_transcript = []
# squash together consecutive lines with the same speaker-id
for ut in whisper_result[1:]:
    if ut.get('speaker', 'none1') == cur_line.get('speaker', 'none2'):
        cur_line = {'start':cur_line['start'], 'end':ut['end'], 'text': cur_line['text'] + ' ' + ut['text'], 'speaker': cur_line['speaker']}
    else:
        audio_transcript.append(cur_line)
        cur_line = ut
audio_transcript.append(cur_line)
at_fpath = f'data/audio_transcripts/{ARGS.vid_name}.json'
breakpoint()
with open(at_fpath, 'w') as f:
    json.dump(audio_transcript, f)


