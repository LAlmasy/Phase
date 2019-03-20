import cv2
import os

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

image_folder = os.path.join(ROOT_DIR, 'results/seq0_merge_tracked/')
video_name = os.path.join(ROOT_DIR, 'results/seq0_merge_tracked.avi')

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") and not img.startswith('.')])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 15, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
