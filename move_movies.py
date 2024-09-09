import os
from os.path import join
from difflib import SequenceMatcher
import json


#downloads_dir = '/home/louis/Downloads'
#movies_with_spaces = [join(downloads_dir, root, fn) for root, dirnames, fnames in os.walk(downloads_dir) for fn in fnames if any(fn.endswith(ext) for ext in ['.mp4', '.mkv', '.avi'])]
#print(f'found {len(movies_with_spaces)} titles in Downloads to move')
#for mws in movies_with_spaces:
#    target = join('torrented-vids', os.path.basename(mws))
#    os.rename(mws, target)
#

#Marcel the Shell
#Ali
#Charade
#Chasing Sleep
#Nothing but a Man - can't find
#Novitiate - can't find
#Parenthood - too big
#The Village - too big
#Thirteen Days - can't find proper version that isn't huge

with open('moviesumm_testset_names.txt') as f:
    clean_names = [x.split('_') for x in f.read().split('\n')]

def cannon_name(vid_name_):
    return best_match

def match_score(x, y):
    x, y = x.lower(), y.lower()
    sm = SequenceMatcher(None, x, y)
    #lcss = sum(x.size for x in sm.get_matching_blocks())
    #return lcss/min(len(x), len(y))
    score = sm.ratio()
    if x in y or y in x:
        score += (1-score)/2 # arbitrary hacky thing
    return score


clean_to_dirty = {}
clean_to_cl = {}
torrented_fpaths = [join('torrented-vids', x) for x in os.listdir('torrented-vids')]
for tfp in torrented_fpaths:
    tpn = os.path.basename(tfp)
    vid_name_orig, _, ext = tpn.rpartition('.')
    vid_name = vid_name_orig.replace('.', ' ').replace('_',' ')
    no_date_list = []
    date = None
    if vid_name == '2012 (2009)':
        date = 2009
        best_match = '2012'
    elif vid_name == 'Mother and Child 2010 1080p BrRip x264 BOKUTOX YIFY':
        date = 2009
        best_match = 'Mother and Child'
    elif vid_name == 'Marcel the Shell with Shoes On 2022 1080p WEB-DL DD5 1 x264-EVO':
        date = 2021
        best_match = 'Marcel the Shell with Shoes On'
    elif vid_name == 'Tin Cup  (Comedy 1996)  Kevin Costner, Rene Russo & Don Johnson':
        date = 1996
        best_match = 'Tin Cup'
    elif vid_name == 'NYAD 2023 1080p WEBRip 1400MB DD5 1 x264-GalaxyRG':
        date = 2023
        best_match = 'Nyad'
    elif vid_name == 'Ali (2001) CE (1080p BluRay x265 HEVC 10bit AAC 5':
        date = 2001
        best_match = 'Ali'
    elif vid_name == 'V For Vendetta 2006 1080p BrRip x264 YIFY':
        date = 2006
        best_match = 'V for Vendetta'
    else:
        for w in vid_name.split():
            if w.lower() in ['bluray', '1080p', 'yify', '720p']:
                continue
            if w.startswith('(') and w.endswith(')'):
                w = w[1:-1]
            try:
                wint = int(w)
                if 1950 < wint < 2024:
                    date = wint
                    continue
            except ValueError:
                pass
            no_date_list.append(w)
        vid_name = ' '.join(no_date_list)
        if vid_name == '':
            breakpoint()
        possible_matches = [x[0] for x in clean_names] if date is None else [x for x,y in clean_names if int(y)==date]
        best_match = max(possible_matches, key=lambda x: match_score(vid_name, x))
    if date is None and best_match not in ['Taxi Driver', "A Hard Day's Night", 'Halloween 4: The Return of Michael Myers']:
        breakpoint()
    if best_match in clean_to_dirty.keys():
        breakpoint()
    clean_to_dirty[best_match] = tpn
    to_change_to = best_match.replace(' ','-').replace(':','').replace('.','').lower()
    clean_to_cl[best_match] = to_change_to
    print(f'renaming {tpn}  -->  {to_change_to}')
    os.rename(tfp, join('torrented-vids', f'{to_change_to}_{date}.{ext}'))

with open('clean-vid-names-to-torrented-names.json', 'w') as f:
    json.dump(clean_to_dirty, f)

with open('clean-vid-names-to-command-line-names.json', 'w') as f:
    json.dump(clean_to_cl, f)

#small_vids = [x for x in torrented_fpaths if os.path.getsize(x) < 700000000]
#print(small_vids)

