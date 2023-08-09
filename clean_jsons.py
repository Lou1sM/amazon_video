import json
import os

json_fnames = [fn for fn in os.listdir('SummScreen') if fn.endswith('.json.safe')]
for fn in json_fnames:
    infpath = os.path.join('SummScreen',fn)
    with open(infpath) as f:
        d=[json.loads(x.strip()) for x in f.readlines()]
    outfpath = infpath[:-5] # ignore the '.safe' suffix
    with open(outfpath,'w') as f:
        json.dump(d,f)
