import json
from nelly_rouge import nelly_rouge
from os.path import join


def episode_from_ep_name(ep_name):
    with open(f'SummScreen/transcripts/{ep_name}.json') as f:
        transcript_data = json.load(f)
    with open(f'SummScreen/summaries/{ep_name}.json') as f:
        summary_data = json.load(f)
    return Episode(ep_name, transcript_data, summary_data)

class Episode(): # Nelly stored transcripts and summaries as separate jsons
    def __init__(self,ep_name,transcript_data,summary_data):
        self.transcript = transcript_data['Transcript']
        self.ep_name = ep_name
        #recap_summ = transcript_data['Recap'][0]
        #trans_summ = transcript_data['Episode Summary'][0]
        #self.summaries = dict(summary_data,recap=recap_summ,trans_summ=trans_summ)
        self.summaries = summary_data
        self.summaries = {k:v for k,v in self.summaries.items() if len(v) > 0}
        self.title = transcript_data['Show Title'].lower().replace(' ','_')
        self.show_name = ep_name.split('.')[0]
        self.transcript_data_dict = transcript_data
        self.summary_data_dict = summary_data

        self.scenes = '\n'.join(self.transcript).split('[SCENE_BREAK]')
        #self.scenes = []
        #prev = ''
        #scenes_with_maybe_emptys = '\n'.join(self.transcript).split('[SCENE_BREAK]')
        #scenes_with_maybe_emptys = [x for x in scenes_with_maybe_emptys if ':' in x or '[' in x]
        #for s in scenes_with_maybe_emptys:
        #    if ':' in s:
        #        self.scenes.append(prev+s)
        #        prev = ''
        #    else:
        #        if len(self.scenes)>0:
        #            self.scenes[-1]+=s
        #        else:
        #            prev = s
        #if any([':' not in x for x in scenes_with_maybe_emptys]):
            #breakpoint()

    def calc_rouge(self,summ):
        best_scores = [-1,-1,-1]
        for summ_name, gt_summ in self.summaries.items():
            print('\n'+summ_name)
            print(gt_summ)
            scores = nelly_rouge(summ,gt_summ)
            if scores[1] > best_scores[1]:
                best_scores = scores
        return best_scores

    def print_recap(self):
        for summ in self.summaries:
            for line in summ.split(' . '):
                print(line)
            print()

    def __repr__(self):
        return f'Episode object for {self.title}'

    def print_transcript(self):
        for line in self.transcript:
            print(line)


class SSEpisode(): # orig SummScreen stored transcripts and summaries as a single json
    def __init__(self,data_dict):
        self.transcript = data_dict['Transcript']
        self.summaries = data_dict['Recap']
        self.title = data_dict['filename'][:-5]
        self.show_name = self.title.split('-')[0]
        self.data_dict = data_dict

        self.scenes = '\n'.join(self.transcript).split('[SCENE_BREAK]')

    def print_recap(self):
        for summ in self.summaries:
            for line in summ.split(' . '):
                print(line)
            print()

    def __repr__(self):
        return f'Episode object for {self.title}'

    def print_transcript(self):
        for line in self.transcript:
            print(line)
