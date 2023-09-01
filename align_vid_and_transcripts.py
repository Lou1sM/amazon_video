import numpy as np
from dl_utils.misc import check_dir
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import json
from nltk import word_tokenize
from difflib import SequenceMatcher
from dtw import dtw
import argparse

def clean(line):
    if ':' not in line:
        return line
    else:
        return line.split(':')[1].lower().strip()

def cc_clean(line):
    return line.replace('[ __ ] ','').strip()

def dist_func(a,b):
    return 1 if a=='' or b=='' else 1-SequenceMatcher(a=a,b=b).ratio()

def align(xlines,ylines):
    sm = SequenceMatcher()
    for xl in xlines:
        if len(xl)==0:
            dist_mat_.append([1]*len(ylines))
        else:
            sm.set_seq2(xl)
            new = []
            for yl in ylines:
                if len(yl)==0:
                    new.append(1)
                else:
                    sm.set_seq1(yl)
                    ratio = sm.find_longest_match()[2]/min(len(xl),len(yl))
                    new.append(1 - ratio)
            dist_mat_.append(new)

    dist_mat = np.stack(dist_mat_)

    alignment = dtw(dist_mat)
    return alignment


parser = argparse.ArgumentParser()
parser.add_argument('-t','--is_test',action='store_true')
parser.add_argument('--db_failed_scenes',action='store_true')
parser.add_argument('--print_full_aligned',action='store_true')
parser.add_argument('--ep_name',type=str, default='oltl-10-18-10')
ARGS = parser.parse_args()


with open(f'SummScreen/transcripts/{ARGS.ep_name}.json') as f:
    transcript_data = json.load(f)

with open(f'SummScreen/closed_captions/{ARGS.ep_name}.json') as f:
    closed_captions = json.load(f)

raw_transcript_lines = transcript_data['Transcript']
transcript_lines = [word_tokenize(clean(line)) for line in raw_transcript_lines]
cc_lines = [word_tokenize(cc_clean(x[1])) for x in closed_captions['captions']]
cc_timestamps = [x[0] for x in closed_captions['captions']]
if ARGS.is_test:
    transcript_lines = transcript_lines[:40]
    cc_lines = cc_lines[:40]
    cc_timestamps = cc_timestamps[:40]
all_words, counts = np.unique(sum(cc_lines+transcript_lines,[]),return_counts=True)
word_to_count = dict(zip(all_words,counts))
N = len(all_words)

video_fpath = f'SummScreen/videos/{ARGS.ep_name}.mp4'
dist_mat = np.array([[dist_func(a,b) for a in cc_lines] for b in transcript_lines])
dist_mat_ = []
alignment = align(transcript_lines, cc_lines)

if ARGS.print_full_aligned:
    for i,j in zip(alignment.index1,alignment.index2):
        print(transcript_lines[i],cc_lines[j], cc_timestamps[j])

def secs_from_timestamp(timestamp):
    hrs,mins,secs_ = timestamp.split(':')
    secs, msecs = secs_.split(',')
    return 3600*float(hrs) + 60*float(mins) + float(secs) + 1e-3*float(msecs)

timestamped_lines = []
starttime = 0
endtime = 0
cur_idx = 0
check_dir(f'SummScreen/video_scenes/{ARGS.ep_name}')
scene_num = 0
scene_starttime = 0
scene_endtime = 0
for idx1, idx2 in zip(alignment.index1,alignment.index2):
    new_starttime, new_endtime = cc_timestamps[idx2].split(' --> ')
    new_starttime = secs_from_timestamp(new_starttime)
    new_endtime = secs_from_timestamp(new_endtime)
    if idx1!=cur_idx or (idx1==alignment.index1.max() and idx2==alignment.index2.max()): # increment transcript lines
        timestamped_tline = f'{starttime} --> {endtime} {raw_transcript_lines[cur_idx]}'
        timestamped_lines.append(timestamped_tline)
        print(timestamped_tline)
        if raw_transcript_lines[cur_idx] == '[SCENE_BREAK]' or (idx1==alignment.index1.max() and idx2==alignment.index2.max()): # increment scenes too
            outpath = f'SummScreen/video_scenes/{ARGS.ep_name}/{ARGS.ep_name}_scene{scene_num}.mp4'
            scene_endtime = min(new_starttime,endtime)
            scene_endtime -= (scene_endtime - scene_starttime)/20 # cut last 5% to reduce overspill
            print(f'SCENE{scene_num}: {scene_starttime}-{scene_endtime}')
            if scene_starttime >= scene_endtime and ARGS.db_failed_scenes:
                breakpoint()
            ffmpeg_extract_subclip(video_fpath,scene_starttime, scene_endtime, targetname=outpath)
            scene_num += 1
            scene_starttime = max(new_starttime,endtime) # start of next scene should be greater than both start of first caption in the next scene and end of last caption in this scene
        assert len(timestamped_lines) == cur_idx+1
        cur_idx = idx1
        starttime = new_starttime
        endtime = new_endtime
    if new_starttime < starttime:
        starttime = new_starttime
        print(777)
    if new_endtime < endtime:
        print(888)
    if new_endtime > endtime:
        endtime = new_endtime

