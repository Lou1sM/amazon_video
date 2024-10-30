name2annot_fn = {
    'td': '1000_Days_scenes.txt',
    'ed': 'ED_1024_scenes.txt',
    'lm': 'Lord_Meia_scenes.txt',
    'r66': 'Route_66_scenes.txt',
    'tos': 'tears_of_steel_1080p_scenes.txt',
    'bbb': 'big_buck_bunny_480p_stereo_scenes.txt',
    'fbw': 'Fires_Beneath_Water_scenes.txt',
    'mer': 'Meridian_scenes.txt',
    'sdm': 'Seven_Dead_Men_scenes.txt',
    'bns': 'Boy_Who_Never_Slept_scenes.txt',
    'hon': 'Honey_scenes.txt',
    'oc': 'Oceania_scenes.txt',
    'sint': 'sintel-1024-surround_scenes.txt',
    'valk': 'Valkaama_1080_p_scenes.txt',
    'ch7': 'CH7_scenes.txt',
    'jw': 'Jathias_Wager_scenes.txt',
    'ssb': 'Sita_Sings_the_Blues_scenes.txt',
    'cl': 'Cosmos_Laundromat_-_First_Cycle_(1080p)_scenes.txt',
    'lcp': 'La_Chute_dune_Plume_scenes.txt',
    'pent': 'Pentagon_scenes.txt',
    'sw': 'Star_Wreck_scenes.txt',
    }

# commented out fps means its in the mp4 file header but the resulting scene
# split times don't look right
vidname2fps = {
    'td': 25,
    'ed': 24,
    'lm': 'UNK FPS', #30,
    'r66': 'UNK FPS',
    'tos': 24,
    'bbb': 30,
    'fbw': 'MISSING VIDEO',
    'mer': 'MISSING VIDEO',
    'sdm': 'UNK FPS', #30,
    'bns': 'MISSING VIDEO',
    'hon': 30,
    'oc': 24,
    'sint': 24,
    'valk': 25,
    'ch7': 'MISSING VIDEO',
    'jw': 30,
    'ssb': 'UNK FPS', #24,
    'cl': 24,
    'lcp': 'UNK FPS', #25,
    'pent': 'MISSING VIDEO',
    'sw': 25,
    }

def get_scene_split_times(vidname):
    fps = vidname2fps[vidname]
    if isinstance(fps, str):
        return fps
    annot_fn = name2annot_fn[vidname]
    annot_path = f'data/annots-osvd/{annot_fn}'

    with open(annot_path) as f:
        scene_lines = f.read().split('\n')

    ssts = [(float(sl.split('\t')[0])/fps,float(sl.split('\t')[1])/fps) for sl in scene_lines]
    return ssts
