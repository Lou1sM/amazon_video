import numpy as np
from fastdtw import fastdtw
import json
from os.path import join
from nltk import word_tokenize
from difflib import SequenceMatcher
from dtw import dtw


ep_fname = 'oltl-10-18-10.json'

with open(join('SummScreen/transcripts',ep_fname)) as f:
    transcript_data = json.load(f)

with open(join('SummScreen/closed_captions',ep_fname)) as f:
    closed_captions = json.load(f)


def old_dist_func(cc,transc):
    return np.maximum(0,cc-transc).sum()

def clean(line):
    if line.startswith('[') and line.endswith(']'):
        return line[1:-1]
    else:
        return line.replace('[','').replace(']','').split(':')[1].lower().strip()

def cc_clean(line):
    return line.replace('[ __ ] ','').strip()

def dist_func(a,b):
    return 1 if a=='' or b=='' else 1-SequenceMatcher(a=a,b=b).ratio()

transcript_lines = [word_tokenize(clean(line)) for line in transcript_data['Transcript']][:40]
cc_lines = [word_tokenize(cc_clean(x[1])) for x in closed_captions['captions']][:40]
cc_timestamps = [x[0] for x in closed_captions['captions']][:40]
all_words, counts = np.unique(sum(cc_lines+transcript_lines,[]),return_counts=True)
word_to_count = dict(zip(all_words,counts))
N = len(all_words)

dist_mat = np.array([[dist_func(a,b) for a in cc_lines] for b in transcript_lines])
dist_mat_ = []
sm = SequenceMatcher()
for tline in transcript_lines:
    if len(tline)==0:
        dist_mat_.append([1]*len(cc_lines))
    else:
        sm.set_seq2(tline)
        new = []
        for cl in cc_lines:
            sm.set_seq1(cl)
            new.append(1 - sm.ratio())
        dist_mat_.append(new)

dist_mat = np.stack(dist_mat_)

alignment = dtw(dist_mat)
for i,j in zip(alignment.index1,alignment.index2): print(transcript_lines[i],cc_lines[j])

def secs_from_timestamp(timestamp):
    hrs,mins,secs_ = timestamp.split(':')
    secs, msecs = secs_.split(',')
    return 3600*float(hrs) + 60*float(mins) + float(secs) + 1e-3*float(msecs)

timestamped_lines = []
starttime = 0
endtime = 0
cur_idx = 0
for idx1, idx2 in zip(alignment.index1,alignment.index2):
    if idx1!=cur_idx:
        assert idx1==cur_idx+1
        timestamped_lines.append(f'{starttime} --> {endtime} {transcript_lines[cur_idx]}')
        assert len(timestamped_lines) == cur_idx+1
        cur_idx = idx1
    new_starttime, new_endtime = cc_timestamps[idx2].split(' --> ')
    new_starttime = secs_from_timestamp(new_starttime)
    new_endtime = secs_from_timestamp(new_endtime)
    if new_starttime < starttime:
        starttime = new_starttime
        print(777)
    if new_endtime > endtime:
        endtime = new_endtime
    else:
        print(888)

#def featify(words_list):
    #return sum([word_to_id[w] for w in words_list])

#word_to_id = {w:(np.arange(N)==i).astype(int)/word_to_count[w] for i,w in enumerate(all_words)}
#base = np.zeros(N)
#transcript_feat_vec_lines = [sum([word_to_id[w] for w in line], base) for line in transcript_lines]
#cc_feat_vec_lines = [sum([word_to_id[w] for w in line], base) for line in cc_lines]
#distance, path = fastdtw(transcript_feat_vec_lines, cc_feat_vec_lines, dist=dist_func)

