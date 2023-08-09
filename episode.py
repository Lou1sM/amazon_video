

class Episode():
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
