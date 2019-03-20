import sys
import numpy as np
import cv2
import os
from tqdm import tqdm
import skimage.morphology
import matplotlib.pyplot as plt

############################################################
#  Combine neighbour cells
############################################################

def combine_neighbour_set(to_merge):
    fusion = False
    for it, m in enumerate(to_merge):
        for n in to_merge[it+1:]:
            if len(m & n)>0:
                to_merge.remove(m)
                to_merge.remove(n)
                to_merge.append(n|m)
                fusion = True
                break
        if fusion:
            break
    if fusion:
        return combine_neighbour_set(to_merge)
    else:
        return to_merge


def merge_neighbours(mask):
    ids = np.unique(mask)[1:]
    to_merge = list()
    for it, current_id in enumerate(ids):
        current = (mask==current_id)
        dilation = skimage.morphology.binary_dilation(current,\
            [[0,1,0],[1,1,1],[0,1,0]])
        size = np.sum(current)
        for other_id in ids[it+1:]:
            other = (mask==other_id)
            intersect = np.logical_and(other, dilation).astype(int)
            overlap = np.sum(intersect)
            if overlap>0:
                if overlap>0.1*size or overlap>0.1*np.sum(other):
                    to_merge.append({current_id, other_id})

    to_merge = combine_neighbour_set(to_merge)

    merged_ids = set([])
    for id_set in to_merge:
        merged_ids = merged_ids | id_set

    lut = np.zeros(ids[-1]+1)
    new_id = 1
    for id in ids:
        if id not in merged_ids:
            lut[id] = new_id
            new_id += 1

    for id_set in to_merge:
        for id in id_set:
            lut[id] = new_id
        new_id += 1

    return lut[mask].astype(int)

############################################################
#  Track cells between frames
############################################################

def make_mask(mask, prev_mask):
    cell_ids = list(np.unique(mask)[1:])
    available = list(range(1, cell_ids[-1]+1))
    lut = np.zeros(cell_ids[-1]+1)

    for prev_cell_id in np.unique(prev_mask)[1:]:
        if prev_cell_id not in available: break
        max_overlap = 0
        cell = np.zeros(mask.shape)
        best_id = 0
        for cell_id in cell_ids:
            intersect = np.logical_and(prev_mask == prev_cell_id, mask==cell_id).astype(int)
            overlap = np.sum(intersect)
            if max_overlap < overlap:
                max_overlap = overlap
                best_id = cell_id
        if best_id != 0:
            cell_ids.remove(best_id)
            lut[best_id]=prev_cell_id
            available.remove(prev_cell_id)

    # Assign all remaining ids to the cells that where not assigned
    # to the same id as a cell from the previous frame
    for cell_id, available_id in zip(cell_ids, available):
        lut[cell_id] = available_id

    new_mask = lut[mask]

    return new_mask


def track_cells(data_dir, submit_dir):
    masks = list()
    mask_ids = sorted(os.listdir(data_dir))
    prev_mask = cv2.imread(os.path.join(data_dir, mask_ids[5]),0)
    prev_mask = merge_neighbours(prev_mask)
    cv2.imwrite(os.path.join(submit_dir, mask_ids[0]), prev_mask)
    for it in tqdm(range(1,len(mask_ids))):
        mask = cv2.imread(os.path.join(data_dir, mask_ids[it]),0)
        mask = merge_neighbours(mask)
        new_mask = make_mask(mask, prev_mask)
        cv2.imwrite(os.path.join(submit_dir, mask_ids[it]), new_mask)
        prev_mask = new_mask

############################################################
#  Command Line
############################################################

# Result directory of the project
RESULTS_DIR = os.path.abspath("../results/")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Provide the mask directory as an argument',\
            '(relative to the results directory)')
        exit()

    data_dir = os.path.join(RESULTS_DIR, sys.argv[1])

    if not os.path.isdir(data_dir):
        print('Could not find directory:', data_dir)
        exit()

    submit_dir = data_dir + "_tracked"
    if os.path.isdir(submit_dir):
        print('Submit directory already exists:', submit_dir)
        exit()

    os.makedirs(submit_dir)

    track_cells(data_dir, submit_dir)
