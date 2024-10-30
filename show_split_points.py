from osvd_names import get_scene_split_times
from dl_utils.misc import time_format
import sys

vidname = sys.argv[1]
ssts = get_scene_split_times(vidname)
if isinstance(ssts, str):
    sys.exit(ssts)

for i, (start, end) in enumerate(ssts):
    print(f'Scene {i}: {time_format(start)} - {time_format(end)}')
