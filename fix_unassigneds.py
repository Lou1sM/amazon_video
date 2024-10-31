import json


with open('moviesumm_testset_names.txt') as f:
    official_names = f.read().split('\n')
with open('clean-vid-names-to-command-line-names.json') as f:
    clean2cl = json.load(f)
#assert all([x in [y.split('_')[0] for y in official_names] for x in clean2cl.keys()])
assert all(x in official_names for x in clean2cl.keys())
test_vidnames = list(clean2cl.values())

for vn in test_vidnames:
    print(vn)
    with open(f'data/transcripts/{vn}-no-names.json') as f:
        noname_transcript = json.load(f)['Transcript']

    with open(f'data/transcripts/{vn}.json') as f:
        wrongname_transcript = json.load(f)['Transcript']

    speaker2char = {}
    for x,y in zip(noname_transcript, wrongname_transcript):
        if x == '[SCENE_BREAK]':
            continue
        speaker_id = x.split(': ')[0]

        if speaker_id == 'UNK':
            assigned_name = 'Unknown Speaker'
        elif y.split(': ')[0]=='UNASSIGNED':
            assigned_name = speaker_id
        else:
            assigned_name = y.split(': ')[0]
        speaker2char[speaker_id] = assigned_name

    print(speaker2char)
    with open(f'data/speaker_char_mappings/{vn}.json', 'w') as f:
        json.dump(speaker2char, f)

    transcript = noname_transcript
    for k,v in speaker2char.items():
        transcript = [line.replace(k, v) for line in transcript]
    with_names_tdata = {'Show Title': vn, 'Transcript': transcript}

    with open(f'data/transcripts/{vn}.json', 'w') as f:
        json.dump(with_names_tdata, f)

