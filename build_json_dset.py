import json
import os
from random import shuffle


scene_summ_dir = 'SummScreen/scene_summs'
summ_dir = 'SummScreen/summaries'
ep_names = os.listdir(scene_summ_dir)
assert all([x.endswith('.txt') for x in ep_names])
print(ep_names)
ep_names = [x[:-4] for x in ep_names]
print(ep_names)
data_list = []
for ep_name in ep_names:
    with open(os.path.join(scene_summ_dir,f'{ep_name}.txt')) as f:
        x = f.read()
    with open(os.path.join(summ_dir,f'{ep_name}.json')) as f:
        d = json.load(f)
    for k,v in d.items():
        if len(v) > 0:
            data_list.append({'scene_summs':x, 'summ':v, 'summ_name':k, 'ep_name':ep_name})

shuffle(data_list)
train_up_to_idx = int(3*len(data_list)/4)
dset = {'train':data_list[:train_up_to_idx], 'test':data_list[train_up_to_idx:]}
with open('SummScreen/scene_summs_to_summ.json','w') as f:
    #json.dump(dset,f)
    json.dump(data_list,f)
