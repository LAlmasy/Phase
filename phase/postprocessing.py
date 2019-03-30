import os
import sys
import cv2
from tqdm import tqdm

from utils.neighbours import merge_neighbours
from utils.track import make_mask
from utils.argparsers import parse_post_processing_args


############################################################
#  Command Line
############################################################

'''
Load mask images and apply the merging of neighbours and the tracking of cells
'''
def track_cells(data_dir, submit_dir, merging=True, tracking=True):
    mask_ids = sorted(os.listdir(data_dir))

    # First mask
    prev_mask = cv2.imread(os.path.join(data_dir, mask_ids[0]),0)
    if merging: prev_mask = merge_neighbours(prev_mask)
    cv2.imwrite(os.path.join(submit_dir, mask_ids[0]), prev_mask)

    # All the next masks
    for it in tqdm(range(1,len(mask_ids))):
        mask = cv2.imread(os.path.join(data_dir, mask_ids[it]),0)

        if merging: mask = merge_neighbours(mask)
        if tracking: new_mask = make_mask(mask, prev_mask)
        else: new_mask = mask

        cv2.imwrite(os.path.join(submit_dir, mask_ids[it]), new_mask)
        prev_mask = new_mask

# Result directory of the project
RESULTS_DIR = os.path.abspath("../results/")

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_post_processing_args()
    # args are dir, merge, track

    if not args.merge and not args.track:
        print('At least one option between --merge and --track must be used.')
        exit()

    data_dir = os.path.join(RESULTS_DIR, args.dir)

    if not os.path.isdir(data_dir):
        print('Could not find directory:', data_dir)
        exit()

    submit_dir = data_dir + "_pp"
    if os.path.isdir(submit_dir):
        print('Submit directory already exists:', submit_dir)
        exit()

    os.makedirs(submit_dir)

    track_cells(data_dir, submit_dir, args.merge, args.track)
