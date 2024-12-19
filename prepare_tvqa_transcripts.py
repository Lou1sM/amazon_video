import os
import json
import re


with open('tvqa_preprocessed_subtitles.json') as f:
    all_clips = json.load(f)

def proc_sub_line(t):
    line = f'{t["text"]}   T: {t["start"]} - {t["end"]}'
    line = line.strip()
    if not bool(re.match(r'[A-Za-z]+ :', line)):
       line = 'UNK: ' + line
    return line


for show_name in os.listdir('data/full-videos/tvqa'):
    for season in os.listdir(os.path.join('data/full-videos/tvqa', show_name)):
        season_num = int(season.removeprefix('season_'))
        for ep in os.listdir(os.path.join('data/full-videos/tvqa', show_name, season)):
            ep_num = int(ep.removeprefix('episode_').removesuffix('.mp4'))
            ep_id = f'{show_name}_s{season_num:02}e{ep_num:02}'
            epclips = [x for x in all_clips if x['vid_name'].startswith(ep_id)]
            epclips = sorted(epclips, key=lambda x:x['vid_name'])
            transcript = '\n[SCENE_BREAK]\n'.join('\n'.join(proc_sub_line(t) for t in x['sub']) for x in epclips)
            tlines = transcript.split('\n')
            with open(f'data/tvqa-transcripts/{ep_id}.json', 'w') as f:
                json.dump({'Show Title': ep_id, 'Transcript': tlines}, f)


