import json
import shutil
import numpy as np
#from utils import get_all_testnames
from datasets import load_dataset
from difflib import SequenceMatcher
from align_vid_and_transcripts import align
import re
from tqdm import tqdm
from lingua import Language, LanguageDetectorBuilder
import argparse


def get_all_testnames():
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

parser = argparse.ArgumentParser()
parser.add_argument('--only-fix-transcripts', action='store_true')
ARGS = parser.parse_args()

test_vidnames, clean2cl = get_all_testnames()
cl2clean = {v:k for k,v in clean2cl.items()}

ds = load_dataset("rohitsaxena/MovieSum")

global_n_correct = 0
global_denom = 0
global_n_correct_baseline_rand = 0
global_denom_baseline_rand = 0
global_n_correct_baseline_most_common = 0
global_denom_baseline_most_common = 0
vidwise_accs = []
vidwise_accs_baseline_rand = []
vidwise_accs_baseline_most_common = []
test_vidnames = ['the-sixth-sense_1999'] + test_vidnames

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
detector = LanguageDetectorBuilder.from_languages(*languages).build()

for vn in (pbar:=tqdm(test_vidnames)):
    #if vn!='somethings-gotta-give_2003':
        #continue
    with open(f'data/transcripts/{vn}-no-names.json') as f:
        raw_transcript_dict = json.load(f)
        raw_transcript = raw_transcript_dict['Transcript']

    with open(f'data/speaker_char_mappings/{vn}.json') as f:
        speaker2char = json.load(f)

    baseline_rand = 1/len(speaker2char)
    baseline_most_common = max(len([tl for tl in raw_transcript if k in tl]) for k in speaker2char.keys())
    global_denom_baseline_rand += len(raw_transcript)
    global_n_correct_baseline_rand += baseline_rand*len(raw_transcript)
    global_denom_baseline_most_common += len(raw_transcript)
    global_n_correct_baseline_most_common += baseline_most_common
    vidwise_accs_baseline_rand.append(baseline_rand)
    vidwise_accs_baseline_most_common.append(baseline_most_common/len(raw_transcript))
    explicit_names = [line for line in raw_transcript[:int(len(raw_transcript)*0.75)] if 'my name is' in line.lower() or 'my name\'s' in line.lower() or re.search(r'(I am|I\'m) [A-Z](?![a-z]*\'s)', line)]
    found_names = []
    for en in explicit_names:
        print('name in line\n', en)
        speaker_id = en.split(':')[0]
        for maybe_pattern in ('my name is', 'my name\'s', "i'm", 'i am'):
            print('searching for pattern', maybe_pattern)
            start_points = [i for i in range(len(en)) if en.lower()[i:].startswith(maybe_pattern)]
            name = ''
            for sp in start_points:
                #name_onwards = en[en.lower().index(maybe_pattern)+len(maybe_pattern):].replace('...','').strip()
                name_onwards = en[sp+len(maybe_pattern):].replace('...','').strip()
                for w in name_onwards.split():
                    if w[0].isupper():
                        name += ' '+w
                    if '.' in w and w not in ('Dr.','Mr.','Mrs','Ms.'):
                        name = name.replace('.','')
                        break
                if name != '':
                    print('found name', name)
                    break
            if name != '':
                break
        if name == '':
            breakpoint()

        #else:
            #declared_name = re.match(r'(?<=(my name is|my name\'s|i\'m|i am).*', en.lower())
            breakpoint()
        name = name.strip()
        found_names.append(name)
        if name == 'Dr Mercer':
            speaker2char[speaker_id] = 'Dr. Julian Mercer'
        elif name == 'Tanya':
            speaker2char[speaker_id] = 'Lexi'
        elif not (len(name)>1 and name.isupper()):
            speaker2char[speaker_id] = name

    #for k,v in speaker2char:


    transcript = [x for x in raw_transcript if x != '[SCENE_BREAK]']
    transcript = [x for x in transcript if set(x.split(':')[1].strip().split()) != set(['you'])]
    transcript = [x for x in transcript if detector.detect_language_of(x) == Language.ENGLISH]
    for k,v in speaker2char.items():
        transcript = [t.replace(k,v) for t in transcript]
    if ARGS.only_fix_transcripts:
        to_dump = dict(raw_transcript_dict, Transcript=transcript)
        outpath = f'data/transcripts/{vn}.json'
        shutil.copy(outpath, outpath+'.safe1')
        with open(outpath, 'w') as f:
            raw_transcript = json.dump(to_dump, f)
        continue

    gt_match_name = cl2clean[vn]
    gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
    gt_script = gt_match['script']
    gt_transcript = []
    speaker_name = ''
    for l in gt_script.split('\n'):
        l = l.strip()
        if l.startswith('<character>'):
            speaker_name = l.removeprefix('<character>').removesuffix('</character>')
        elif l.startswith('<dialogue>'):
            spoken = l.removeprefix('<dialogue>').removesuffix('</dialogue>')
            gt_transcript.append(f'{speaker_name}: {spoken}')

    n_correct = 0
    denom = 0
    #transcript= [line.split(':')[0] + ': ' + x for line in transcript for x in line.split(':')[1].replace('...','').split('.') if x.strip()!='']
    #gt_transcript= [line.split(':')[0] + ': ' + x for line in gt_transcript for x in line.split(':')[1].replace('...','').split('.') if x.strip()!='']
    #alignment = align(transcript_sents, gt_transcript_sents)
    alignment = align(transcript, gt_transcript)
    for i,j in zip(alignment.index1,alignment.index2):
        predline, gtline = transcript[i], gt_transcript[j]
        #print(predline, gtline)
        pred_speaker = predline.split(':')[0]
        if pred_speaker != 'unknown speaker' and 'SPEAKER' not in pred_speaker:
            if SequenceMatcher(None, predline, gtline).ratio() > 0.8:
                denom += 1
                if pred_speaker in found_names:
                    n_correct += 1
                else:
                    gt_speaker = gtline.split(':')[0]
                    if gt_speaker.lower() in pred_speaker.lower():
                        n_correct += 1
    if denom > 0:
        if n_correct==0 and denom>40:
            print('acc zero for', vn)
        vidwise_accs.append(n_correct/denom)
        global_denom += denom
        global_n_correct += n_correct
    pbar.set_description(f'Acc: {np.array(vidwise_accs).mean():.3f}  Global Acc: {global_n_correct/global_denom:.3f}')


results = {
        'acc': np.array(vidwise_accs).mean(),
        'acc-global': global_n_correct/global_denom,
        'acc-baseline-rand': np.array(vidwise_accs_baseline_rand).mean(),
        'acc-global-baseline-rand': global_n_correct_baseline_rand/global_denom_baseline_rand,
        'acc-baseline-most-common': np.array(vidwise_accs_baseline_most_common).mean(),
        'acc-global-baseline-most-common': global_n_correct_baseline_most_common/global_denom_baseline_most_common,
        }

with open('names-results.json', 'w') as f:
    json.dump(results, f)

print(results)
breakpoint()

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
