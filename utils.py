import re
from nltk.tokenize import sent_tokenize
import json
import os
from dl_utils.tensor_funcs import numpyify
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from dl_utils.label_funcs import accuracy
from dl_utils.misc import check_dir
import rouge
import torch
import numpy as np
import pandas as pd
from natsort import natsorted
from functools import partial
from datasets import load_dataset
#from nltk.metrics import windowdiff


metric_names = ['acc','nmi','ari', 'pk', 'winddiff', 'ded']
rouge_eval = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                                 max_n=2,
                                                 limit_length=False,
                                                 apply_avg=True,
                                                 apply_best=False,
                                                 alpha=0.5, # Default F1_score
                                                 stemming=False)

def display_rouges(r):
    return list(zip(['r1','r2','rL','rLsum'],r))

def rouge_preprocess(text):
    text = text.replace('...',' <eplipsis> ')
    text = rouge.Rouge.REMOVE_CHAR_PATTERN.sub(' ', text.lower()).strip()
    tokens = rouge.Rouge.tokenize_text(rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD.sub('_cannot_', text))
    rouge.Rouge.stem_tokens(tokens)
    preprocessed_text = rouge.Rouge.KEEP_CANNOT_IN_ONE_WORD_REVERSED.sub('cannot', ' '.join(tokens))
    return preprocessed_text

def nelly_rouge(pred_,gt_):
    pred_sum_sents = [rouge_preprocess(p) for p in split_summ(pred_)]
    gt_sum_sents = [rouge_preprocess(g) for g in split_summ(gt_)]
    pred = '\n'.join(pred_sum_sents)
    gt = '\n'.join(gt_sum_sents)

    scores = rouge_eval.get_scores(pred, gt)

    pred_old = [rouge_preprocess(pred_)]
    gt_old = [rouge_preprocess(gt_)]
    old_scores = rouge_eval.get_scores(pred_old, gt_old)
    scores['rouge-lsum'] = scores['rouge-l']
    scores['rouge-l'] = old_scores['rouge-l']
    return scores

def old_nelly_rouge(pred,gt):
    if not isinstance(pred,list):
        pred = [pred]
    if not isinstance(gt,list):
        gt = [gt]
    pred_sums = [rouge_preprocess(pred) for pred in pred]
    gt_sums = [rouge_preprocess(g) for g in gt]
    scores = rouge_eval.get_scores(pred_sums, gt_sums)
    return scores

def split_summ(s):
    return s.replace('. ','.\n').split('\n')

def extract_main_rouges(scores):
    rouge1 = scores['rouge-1']['f'] * 100
    rouge2 = scores['rouge-2']['f'] * 100
    rougel = scores['rouge-l']['f'] * 100
    rougelsum = scores['rouge-lsum']['f'] * 100
    return rouge1, rouge2, rougel, rougelsum

def rouge_from_multiple_refs(pred, references, return_full, benchmark_rl):
    benchmark = -1
    for possible_gt in references:
        new_rouge = nelly_rouge(pred, possible_gt)
        maybe_new_benchmark = new_rouge['rouge-l']['f'] if benchmark_rl else new_rouge['rouge-2']['f']
        if maybe_new_benchmark > benchmark:
            benchmark = maybe_new_benchmark
            best_rouge = new_rouge
    if benchmark == 0:
        if not all([gt is None for gt in references]):
            print('rouge is zero')
    return best_rouge if return_full else extract_main_rouges(best_rouge)

def get_fn(caps, order, uniform_breaks, startendscenes, centralscenes, is_test):
    fn = caps
    if order=='optimal':
        fn += '_reordered'
    if order=='rand':
        fn += '_rand_ordered'
    if uniform_breaks:
        fn += '_uniform_breaks'
    if startendscenes:
        fn += '_startendscenes'
    if centralscenes:
        fn += '_centralscenes'
    if is_test:
        fn += '_test'
    #if n_dpoints != -1:
        #fn += f'{n_dpoints}dps'
    return fn

def safe_decode(tokens, tokenizer):
     st = [[x for x in ts[:tokenizer.model_max_length] if x != -100] for ts in tokens]
     return tokenizer.batch_decode(st, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def chunkify(text,max_chunk_size):
    if len(text.split())*4/3 < max_chunk_size:
        to_return = [text]
    else:
        first_chunk, second_chunk = split_text_by_sth(text)
        to_return = chunkify(first_chunk,max_chunk_size) + chunkify(second_chunk,max_chunk_size)
    if not all(len(x) <= max_chunk_size for sl in to_return for x in sl):
        breakpoint()
    return to_return

def split_text_by_sth(text):
    for sep in ('\n', '. ', ', ', ' '):
        if sep in text.strip():
            return split_text_by_sep(text.strip(),sep)
    return text[:len(text)//2], text[len(text)//2:]

def summ_short_scene(text):
    return ' '.join(convert_script_to_prose(line) for line in text.split('\n') if line!='')

def convert_script_to_prose(script_line):
    if maybe_speaker_name:=re.match(r'\w+: ', script_line):
        speaker_name = script_line[:maybe_speaker_name.span()[1]-2]
        speech = script_line[maybe_speaker_name.span()[1]:]
        return f'{speaker_name} said "{speech}"'
    elif stage_direction := re.match(r'(?<=\[ )[A-Z -]+(?= \])', script_line):
        return stage_direction
    else:
        return script_line

def split_text_by_sep(text,sep):
    lines = text.split(sep)
    N = len(text.split())
    first_chunk = ''
    for i,l in enumerate(lines):
        if abs(len((first_chunk+l).split()) - N/2) > abs(len(first_chunk.split())-N/2):
            break # get as close to halfway as possible
        if first_chunk=='':
            first_chunk = l+sep
        else:
            first_chunk += l+sep
        if not text.startswith(first_chunk):
            breakpoint()
    second_chunk = text[len(first_chunk):]
    assert first_chunk+second_chunk == text
    return first_chunk, second_chunk

def prepare_for_pil(torch_im):
    assert isinstance(torch_im, torch.Tensor)
    normed_torch_im = (torch_im - torch_im.min()) / (torch_im.max() - torch_im.min())
    np_im = normed_torch_im.permute(1,2,0).numpy()
    np_uint8_im = (np_im*255).astype(np.uint8)
    return np_uint8_im

def shim(im):
    import matplotlib.pyplot as plt
    plt.imshow(im); plt.show()

def tshim(t):
    a = numpyify(t.permute(1,2,0))
    shim(a)

def get_all_testnames(exclude_non_english=True):
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    if exclude_non_english:
        clean2cl = {k:v for k,v in clean2cl.items() if v not in ['the-girl-with-the-dragon-tattoo_2011', 'austin-powers-international-man-of-mystery_1997']}
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

def path_list(parent_dir):
    return natsorted([os.path.join(parent_dir, child) for child in os.listdir(parent_dir)])

def bernoulli_CE(p1, p2):
    return -p1 * np.log(p2) - (1-p1)*np.log(1-p2)

def is_prisma_wellformed(sent):
    if any(x in sent for x in ['Vs', 'v&h', '(v/Kv)', 'V2', 'V3', 'V4', 'V&S', 'V&H', 'v&S', 'html', '.com', '.co.uk']): # Bart garbage
        return False
    if any(sent.endswith(x) for x in ['the', 'a', 'is']):
        return False
    if len(sent.split())==2 and sent.strip().startswith('The'):
        return False
    if len(sent.split())==1:
        return False
    if any(ord(c)>=128 for c in list(sent)):
        return False
    if ':' in sent or '?' in sent:
        return False
    return True

def postfilter(sent):
    if any(x in sent.split() for x in ['I', 'you', 'we']):
        return False
    if 'please note' in sent.lower():
        return False
    if any(w in sent.split() for w in ['genre', 'classic', 'style', 'director', 'directed']):
        return False
    if 'SPEAKER' in sent:
        return False
    return True

def segmentation_metrics(preds, gt_point_labs, k):
    results = {}
    set_pk = partial(p_k, k=k)
    set_windowdiff = partial(windowdiff, k=k)
    for mname, mfunc in zip(metric_names, [acc, nmi, ari, set_pk, set_windowdiff, ded]):
        score = mfunc(gt_point_labs, preds)
        results[mname] = score
    return results

def acc(preds, gts):
    acc1 = accuracy(preds, gts)
    acc2 = accuracy(gts, preds)
    return (acc1+acc2)/2

def p_k(preds, gts, k):
    scores_by_i = []
    for i in range(1,k+1):
        pred_same_diffs = preds[i:] == preds[:-i]
        gt_same_diffs = gts[i:] == gts[:-i]
        scores_by_i.append((pred_same_diffs==gt_same_diffs).mean())
    return np.array(scores_by_i).mean()

def windowdiff(preds, gts, k):
    assert len(preds)==len(gts)
    wd = 0
    for i in range(len(preds) - k):
        n_pred_boundaries = preds[i+k] - preds[i]
        n_gt_boundaries = gts[i+k] - gts[i]
        wd += abs(n_pred_boundaries - n_gt_boundaries)
    return wd / (len(preds)-k)

def ded(preds, gts):
    if len(np.unique(preds)) < len(np.unique(gts)):
        return 1 - accuracy(preds, gts)
    else:
        return 1 - accuracy(gts, preds)

def get_moviesumm_testnames():
    with open('moviesumm_testset_names.txt') as f:
        official_names = f.read().split('\n')
    with open('clean-vid-names-to-command-line-names.json') as f:
        clean2cl = json.load(f)
    #assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
    assert all(x in official_names for x in clean2cl.keys())
    test_vidnames = list(clean2cl.values())
    return test_vidnames, clean2cl

def bbc_mean_maxs(results_df):
    mean_avgs = results_df.mean(axis=0).unstack().groupby(axis=0, level=0).mean()
    results_df.loc[:, pd.IndexSlice[:, :, 'winddiff']] = -results_df.loc[:, pd.IndexSlice[:, :, 'winddiff']]
    results_df.loc[:, pd.IndexSlice[:, :, 'ded']] = -results_df.loc[:, pd.IndexSlice[:, :, 'ded']]
    max_avgs = results_df.max(axis=0).unstack().groupby(axis=0, level=0).mean()
    max_avgs['winddiff'] = -max_avgs['winddiff']
    max_avgs['ded'] = -max_avgs['ded']
    return max_avgs, mean_avgs

def get_moviesumm_splitpoints(vn):
    from lingua import Language, LanguageDetectorBuilder
    from align_vid_and_transcripts import align
    check_dir('data/moviesumm-gt-splitpoints')
    if os.path.exists(json_fp:=f'data/moviesumm-gt-splitpoints/{vn}.json'):
        with open(json_fp) as f:
            d = json.load(f)
        gt_split_points = np.array(d['split-points'])
        betweens = np.array(d['betweens']).astype(bool)

    else:
        ds = load_dataset("rohitsaxena/MovieSum")
        with open(f'data/whisper_outputs/{vn}.json') as f:
           wlines = json.load(f)
        languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]
        detector = LanguageDetectorBuilder.from_languages(*languages).build()

        test_vidnames, clean2cl = get_moviesumm_testnames()
        cl2clean = {v:k for k,v in clean2cl.items()}

        wlines = [ut for ut in wlines if not set(ut['text'].split())==set(['you'])]
        wlines = [ut for ut in wlines if detector.detect_language_of(ut['text']) == Language.ENGLISH]
        wspoken = [ut['text'] for ut in wlines]

        gt_match_name = cl2clean[vn]
        gt_match = [x for x in ds['test'] if x['movie_name']==gt_match_name][0]
        gt_script = gt_match['script']
        gt_spoken = []
        all_scene_idxs = []
        scene_idx = 0
        for l in gt_script.split('\n'):
            l = l.strip()
            if l.startswith('<scene>'):
                scene_idx += 1
            elif l.startswith('<dialogue>'):
                spoken = l.removeprefix('<dialogue>').removesuffix('</dialogue>')
                gt_spoken.append(spoken)
                all_scene_idxs.append(scene_idx)

        alignment = align(wspoken, gt_spoken)
        assert len(wlines)==len(wspoken)
        assert len(all_scene_idxs)==len(gt_spoken)
        #gt = np.array(all_scene_idxs)[alignment.index2]
        #prev_endtime = 0
        curr_starttime = None
        curr_endtime = 0
        prev_sidx = all_scene_idxs[0]
        all_startends = []
        for i,j in zip(alignment.index1, alignment.index2):
            wut = wlines[i]
            assert wut['end']>=curr_endtime
            if curr_starttime is None:
                curr_starttime = wut['start']
            else:
                assert wut['start']>=curr_starttime

            curr_endtime=wut['end']
            sidx = all_scene_idxs[j]
            if sidx!=prev_sidx:
                all_startends.append((curr_starttime,curr_endtime))
                prev_sidx = sidx
                curr_starttime = None

        all_starts = np.array([x[0] for x in all_startends])
        all_ends = np.array([x[1] for x in all_startends])
        gt_split_points = (all_starts[1:] + all_ends[:-1]) / 2
        kf_timepoints = np.load(f'data/ffmpeg-keyframes/{vn}/frametimes.npy')
        betweens = np.array([any(x>all_startends[i][1] and x<all_startends[i+1][0] for i in range(len(all_startends)-1)) for x in kf_timepoints])

        to_save = {'split-points': list(gt_split_points), 'betweens': [int(x) for x in betweens]}
        with open(json_fp, 'w') as f:
            json.dump(to_save, f)
    return gt_split_points, betweens

def split_points_to_labels(split_points, timepoints_to_label):
    return (np.expand_dims(timepoints_to_label,1)>split_points).sum(axis=1)

def drop_trailing_halfsent(s):
    s = s.replace('Dr.','DRXXX')
    s = s.replace('Lt.','LtXXX')
    s = s.replace('Mr.','MrXXX')
    s = s.replace('Mrs.','MrsXXX')
    s = s.replace('Ms.','MsXXX')
    #s = '. '.join(x for x in s.split('. ')[:-1])
    s = ' '.join(x for x in sent_tokenize(s) if x.endswith('.'))
    s = s.replace('XXX', '.')
    return s

