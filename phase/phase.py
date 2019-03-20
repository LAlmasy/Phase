"""
Mask R-CNN

"""

import os
import sys
import json
import datetime
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Channel selected when loading phase image from multiple directories
INPUT_CHANNEL = 'raw.tif'


############################################################
#  Configurations
############################################################

class PhaseConfig(Config):
    """Configuration for training on the phase contrast segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "phase"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 45
    VALIDATION_STEPS = 5

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    # DETECTION_MIN_CONFIDENCE = 0
    DETECTION_MIN_CONFIDENCE = 0.5 # Usiigaci

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    #BACKBONE = "resnet50"
    BACKBONE = "resnet101" # Usiigaci

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.9
    RPN_NMS_THRESHOLD = 0.99 # Usiigaci

    # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128 # Usiigaci

    # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([126,126,126]) # Usiigaci

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    # MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    MINI_MASK_SHAPE = (100,100) # Usiigaci

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    # TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 256 # Usiigaci

    # Maximum number of ground truth instances to use in one image
    # MAX_GT_INSTANCES = 200
    MAX_GT_INSTANCES = 500 # Usiigaci

    # Max number of final detections per image
    # DETECTION_MAX_INSTANCES = 400
    DETECTION_MAX_INSTANCES = 400 #Usiigaci


class PhaseInferenceConfig(PhaseConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class PhaseDataset(utils.Dataset):

    def load_phase(self, dataset_dir, single_dir=False):
        """Load the dataset.

        dataset_dir: Root directory of the dataset
        single_dir: True if all images are in the same dir
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("cell", 1, "cell")

        image_ids = os.listdir(dataset_dir)

        if not single_dir:
            for image_id in image_ids:
                if image_id.startswith('set'):
                    self.add_image(
                        'cell',
                        image_id=image_id,
                        path=os.path.join(dataset_dir, image_id, INPUT_CHANNEL)
                    )
        else:
            for image_id in image_ids:
                if os.path.splitext(image_id)[1] in ('.tif', '.jpg', '.bmp')\
                    and not image_id.startswith('.'):
                    self.add_image(
                        'cell',
                        image_id=os.path.splitext(image_id)[0],
                        path=os.path.join(dataset_dir, image_id)
                    )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.dirname(info['path'])
        mask_path = os.path.join(mask_dir, 'instances_ids.png')

        # Read mask files from .png image
        ids_mask = cv2.imread(mask_path, 0)
        instances_num = len(np.unique(ids_mask)) - 1
        mask = np.zeros((ids_mask.shape[0], ids_mask.shape[1], instances_num))
        for i in range(instances_num):
            # print(np.where(ids_mask == (i + 1)))
            slice = mask[..., i]
            slice[np.where(ids_mask == (i + 1))] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)

print('Loaded phase configs')
