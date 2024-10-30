import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys


vn = sys.argv[1]

skf_dir = f'data/ffmpeg-keyframes-by-scene/{vn}'
for sdir in os.listdir(skf_dir):
    x = np.array([Image.open(os.path.join(skf_dir, sdir, fn)) for fn in os.listdir(os.path.join(skf_dir, sdir))])
    blurred_im = x.mean(axis=0)/255
    plt.imshow(blurred_im)
    plt.savefig(out_fp:=f'/tmp/all-kfs-{sdir}.png')
    os.system(f'/usr/bin/xdg-open {out_fp}')

