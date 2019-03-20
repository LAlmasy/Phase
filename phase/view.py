import matplotlib.pyplot as plt
import cv2
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

phase = cv2.imread(phase, 0)
mask = cv2.imread(mask, 0)

mask = np.ma.masked_where(mask < 0.9, mask)


plt.figure(figsize=(15,10))

plt.imshow(phase, cmap=plt.cm.gray)
plt.imshow(mask, alpha=0.6)

plt.show()
