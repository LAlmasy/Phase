'''
Visualise a mask over an image.
'''

import matplotlib.pyplot as plt
import skimage.segmentation
import skimage.io
import skimage.color
import os
import numpy as np
from utils.argparsers import parse_view_arguments

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Parse command line arguments
args = parse_view_arguments()
# args are mask, img and borders

mask = os.path.join(ROOT_DIR, args.mask)
phase = os.path.join(ROOT_DIR, args.img)

if not os.path.isfile(mask):
    print('Invalid mask path:', mask)
    exit()
if not os.path.isfile(phase):
    print('Invalid image path:', phase)
    exit()

phase = skimage.io.imread(phase)
mask = skimage.io.imread(mask)

if args.borders:
    marked = skimage.segmentation.mark_boundaries(phase, mask)

    plt.figure(figsize=(15,10))

    plt.imshow(marked)
else:
    colored_mask = skimage.color.label2rgb(mask, bg_label=0)

    colored_mask = np.ma.masked_where(colored_mask == (0,0,0), colored_mask)

    plt.figure(figsize=(15,10))

    plt.imshow(phase, cmap=plt.cm.gray)
    plt.imshow(colored_mask, alpha=0.5)

plt.show()
