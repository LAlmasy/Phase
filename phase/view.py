'''
Visualise a mask over an image.
'''

import matplotlib.pyplot as plt
import skimage.segmentation
import skimage.io
import skimage.color
import os
import numpy as np
import argparse

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Visualisation of an image and its mask')
parser.add_argument('--img', required=True,
                    metavar="/path/to/image.*",
                    help='Path to image')
parser.add_argument('--mask', required=True,
                    metavar="/path/to/mask.*",
                    help="Path to mask")
args = parser.parse_args()

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

boundaries = False
if boundaries:
    marked = skimage.segmentation.mark_boundaries(phase, mask)

    plt.figure(figsize=(15,10))

    plt.imshow(marked)
else:
    colored_mask = m = skimage.color.label2rgb(mask, bg_label=0)

    colored_mask = np.ma.masked_where(colored_mask == (0,0,0), colored_mask)

    plt.figure(figsize=(15,10))

    plt.imshow(phase, cmap=plt.cm.gray)
    plt.imshow(colored_mask, alpha=0.5)

plt.show()
