import os
import subprocess
import json
import imageio_ffmpeg
from tqdm import tqdm


with open('clean-vid-names-to-command-line-names.json') as f:
    official2cl = json.load(f)
vid_names = official2cl.values()

for vn in tqdm(list(vid_names)[23:]):
    if os.path.exists(transc_fpath:=f'data/whisper_outputs/{vn}.json'):
        print(f'Transcription already exists at {transc_fpath}, skipping')
        continue
    audio_fpath = f'data/audio/{vn}.wav'

    if not os.path.exists(audio_fpath):
        mp4_fpath = f'data/full_videos/{vn}.mp4'
        print(f'No file found at {audio_fpath}\nExtracting audio from {mp4_fpath}')
        FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
        extract_wav_cmd = f"{FFMPEG_PATH} -i {mp4_fpath} -ab 160k -ac 2 -ar 44100 -vn {audio_fpath} -loglevel quiet"
        subprocess.call(extract_wav_cmd, shell=True)
