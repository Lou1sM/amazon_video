import json
import re


def is_line_has_speaker_name(line):
    if len(line.split())==1 or ':' not in line:
        return False
    if ':' in line.split()[1:3]:
        return True
    first_chunk = line[line.index(':')]
    first_chunk = re.sub(r'\(.*\)','',first_chunk)
    if len(first_chunk.split()) > 3:
        print(line)
        return False
    else:
        return True

def is_speaker_line(line):
    if line[0] not in ['[','(']:
        return True
    return not (re.match(r'^\(.*\)$',line) or re.match(r'^\[.*\]$',line))

with open('SummScreen/tms_train.json') as f:
    data = json.load(f)

n = 0
n_names = 0
for ep in data:
    speaker_lines = [line for line in ep['Transcript'] if is_speaker_line(line)]
    n_names += sum([is_line_has_speaker_name(line) for line in speaker_lines])
    n += len(speaker_lines)
print(n_names, n, n_names/n)
