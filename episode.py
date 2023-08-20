

class Episode(): # Nelly stored transcripts and summaries as separate jsons
    def __init__(self,ep_fname,transcript_data,summary_data):
        self.transcript = transcript_data['Transcript']
        self.ep_fname = ep_fname
        #recap_summ = transcript_data['Recap'][0]
        #trans_summ = transcript_data['Episode Summary'][0]
        #self.summaries = dict(summary_data,recap=recap_summ,trans_summ=trans_summ)
        self.summaries = summary_data
        self.summaries = {k:v for k,v in self.summaries.items() if len(v) > 0}
        self.title = transcript_data['Show Title'].lower().replace(' ','_')
        self.show_name = ep_fname.split('.')[0]
        self.transcript_data_dict = transcript_data
        self.summary_data_dict = summary_data

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
