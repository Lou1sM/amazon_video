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
    with open(f'data/postprocessed-video-captions/{vn}/kosmos_procced_scene_caps.json') as f:
        caps_with_unks = json.load(f)
    assert not any('UNK' in x['raw'] for x in caps_with_unks)
    caps_without_unks = [{'scene_id': x['scene_id'], 'raw':x['raw'], 'with_names':x['raw'] if 'UNK' in x else x['with_names']} for x in caps_with_unks]
    with open(f'data/postprocessed-video-captions/{vn}/kosmos_procced_scene_caps.json.safe','w') as f:
        json.dump(caps_without_unks, f)

