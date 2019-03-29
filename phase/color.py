import skimage.io
import skimage.color
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
path = os.path.join(ROOT_DIR, "results/report_merge_tracked/exp0167.png")

im = skimage.io.imread(path)
im = skimage.color.label2rgb(im, bg_label=0)
skimage.io.imsave(path[:-4]+'_c.png', im)
