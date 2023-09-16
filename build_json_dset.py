import json
import os
from random import shuffle


scene_summ_dir = 'SummScreen/scene_summs'
summ_dir = 'SummScreen/summaries'
ep_names = os.listdir(scene_summ_dir)
assert all([x.endswith('.txt') for x in ep_names])
ep_names = [x[:-4] for x in ep_names]
shuffle(ep_names)
train_up_to_idx = int(3*len(ep_names)/4)
train_ep_names = ep_names[:train_up_to_idx]
test_ep_names = ep_names[train_up_to_idx:]
print(test_ep_names)
train_data_list = []
for ep_name in train_ep_names:
    with open(os.path.join(scene_summ_dir,f'{ep_name}.txt')) as f:
        x = f.read()
    with open(os.path.join(summ_dir,f'{ep_name}.json')) as f:
        d = json.load(f)
    for k,v in d.items():
        if len(v) > 0 and k!='soap_central':
            train_data_list.append({'scene_summs':x, 'summ':v, 'summ_name':k, 'ep_name':ep_name})

with open('SummScreen/scene_summs_to_summ_train.json','w') as f:
    json.dump(train_data_list,f)

test_data_list = []
for i,ep_name in enumerate(test_ep_names):
    with open(os.path.join(scene_summ_dir,f'{ep_name}.txt')) as f:
        x = f.read()
    with open(os.path.join(summ_dir,f'{ep_name}.json')) as f:
        d = json.load(f)
    dpoint = {'ep_name':ep_name, 'scene_summs':x}
    print(d.keys())
    for k,v in d.items():
        if len(v) > 0 and k!='soap_central':
            dpoint[k] = v
        if len(v) == 0:
            breakpoint()
    if i==0:
        del dpoint['tvmega_recap']
    test_data_list.append(dpoint)

with open('SummScreen/scene_summs_to_summ_test.json','w') as f:
    json.dump(test_data_list,f)
