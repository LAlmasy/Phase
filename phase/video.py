import cv2
import os
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Frames per second
FPS = 15

if len(sys.argv) != 2:
    print('Provide an image directory as an argument',\
        '(relative to the project directory)')
    exit()

image_dir = os.path.join(ROOT_DIR, sys.argv[1])

if not os.path.isdir(image_dir):
    print('Could not find directory:', image_dir)
    exit()

if image_dir.endswith('/'):
    video_name = image_dir[:-1] + '.avi'
else:
    video_name = image_dir + '.avi'

# Create video
images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png") and not img.startswith('.')])
frame = cv2.imread(os.path.join(image_dir, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, FPS, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_dir, image)))

cv2.destroyAllWindows()
video.release()
