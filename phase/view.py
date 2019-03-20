import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

mask = os.path.join(ROOT_DIR, "results/20190314_112705_merge/exp0249.png")
phase = os.path.join(ROOT_DIR, "test_data/exp0249.jpg")

phase = cv2.imread(phase, 0)
mask = cv2.imread(mask, 0)

mask = np.ma.masked_where(mask < 0.9, mask)


plt.figure(figsize=(15,10))

plt.imshow(phase, cmap=plt.cm.gray)
plt.imshow(mask, alpha=0.6)

plt.show()
