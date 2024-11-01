
annot_fns_list = [
         '01_From_Pole_to_Pole.txt',
         '02_Mountains.txt',
         '03_Ice Worlds.txt',
         '04_Great Plains.txt',
         '05_Jungles.txt',
         '06_Seasonal_Forests.txt',
         '07_Fresh_Water.txt',
         '08_Ocean_Deep.txt',
         '09_Shallow_Seas.txt',
         '10_Caves.txt',
         '11_Deserts.txt',
         ]

def get_scene_split_times(vidname):
    fps = 25

    vn = annot_fns_list[int(vidname)]
    with open(f'data/bbc-annotations/shots/{vn}') as f:
        shot_break_frame_nums = [int(x.split('\t')[0]) for x in f.read().strip().split('\n')]

    all_scene_break_times = []
    for annot_num in range(5):
        with open(f'data/bbc-annotations/scenes/annotator_{annot_num}/{vn}') as f:
            scene_break_shot_nums = [int(x) for x in f.read().split(',')]

        annot_scene_break_shot_nums = [shot_break_frame_nums[x] for x in scene_break_shot_nums[1:-1]]
        annot_scene_break_times = [x/fps for x in annot_scene_break_shot_nums]
        all_scene_break_times.append(annot_scene_break_times)

    return all_scene_break_times


if __name__ == '__main__':
    import sys
    from dl_utils.misc import time_format
    vn = sys.argv[1]
    for annot_num, annot in enumerate(get_scene_split_times(vn)):
        print(f'Annotator {annot_num}')
        for i, s in enumerate(annot):
            print(f'Scene{i}: {time_format(s)}')
