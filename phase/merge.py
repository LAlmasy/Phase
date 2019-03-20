import os
from tqdm import tqdm
import cv2
import datetime
import numpy as np

# Result directory of the project
RESULTS_DIR = os.path.abspath("../results/")

############################################################
#  Usiigaci method to merge
############################################################

IOU_THRESHOLD = 0.6
OVERLAP_THRESHOLD = 0.8
MIN_DETECTIONS = 1

def compute_iou(mask1, mask2):
    """
    Computes Intersection over Union score for two binary masks.
    :param mask1: numpy array
    :param mask2: numpy array
    :return:
    """
    intersection = np.sum((mask1 + mask2) > 1)
    union = np.sum((mask1 + mask2) > 0)

    return intersection / float(union)


def compute_overlap(mask1, mask2):
    intersection = np.sum((mask1 + mask2) > 1)

    overlap1 = intersection / float(np.sum(mask1))
    overlap2 = intersection / float(np.sum(mask2))
    return overlap1, overlap2


def sort_mask_by_cells(mask, min_size=50):
    """
    Returns size of each cell.
    :param mask:
    :return:
    """
    cell_num = np.unique(mask)
    cell_sizes = [(cell_id, len(np.where(mask == cell_id)[0])) for cell_id in cell_num if cell_id != 0]

    cell_sizes = [x for x in sorted(cell_sizes, key=lambda x: x[1], reverse=True) if x[1 > min_size]]

    return cell_sizes


def merge_masks(masks):
    """

    :param masks:
    :return:
    """
    cell_counter = 0
    final_mask = np.zeros(masks[0].shape)

    masks_stats = [sort_mask_by_cells(mask) for mask in masks]
    cells_left = sum([len(stats) for stats in masks_stats])

    while cells_left > 0:
        # Choose the biggest cell from available
        cells = [stats[0][1] if len(stats) > 0 else 0 for stats in masks_stats]
        reference_mask = cells.index(max(cells))

        reference_cell = masks_stats[reference_mask].pop(0)[0]

        # Prepare binary mask for cell chosen for comparison
        cell_location = np.where(masks[reference_mask] == reference_cell)

        cell_mask = np.zeros(final_mask.shape)
        cell_mask[cell_location] = 1

        masks[reference_mask][cell_location] = 0

        # Mask for storing temporary results
        tmp_mask = np.zeros(final_mask.shape)
        tmp_mask += cell_mask

        for mask_id, mask in enumerate(masks):
            # For each mask left
            if mask_id != reference_mask:
                # # Find overlapping cells on other masks
                overlapping_cells = list(np.unique(mask[cell_location]))

                try:
                    overlapping_cells.remove(0)
                except ValueError:
                    pass

                # # If only one overlapping, check IoU and update tmp mask if high
                if len(overlapping_cells) == 1:
                    overlapping_cell_mask = np.zeros(final_mask.shape)
                    overlapping_cell_mask[np.where(mask == overlapping_cells[0])] = 1

                    iou = compute_iou(cell_mask, overlapping_cell_mask)
                    if iou >= IOU_THRESHOLD:
                        # Add cell to temporary results and remove from stats and mask
                        tmp_mask += overlapping_cell_mask
                        idx = [i for i, cell in enumerate(masks_stats[mask_id]) if cell[0] == overlapping_cells[0]][0]
                        masks_stats[mask_id].pop(idx)
                        mask[np.where(mask == overlapping_cells[0])] = 0

                # # If more than one overlapping check area overlapping
                elif len(overlapping_cells) > 1:
                    overlapping_cell_masks = [np.zeros(final_mask.shape) for _ in overlapping_cells]

                    for i, cell_id in enumerate(overlapping_cells):
                        overlapping_cell_masks[i][np.where(mask == cell_id)] = 1

                    for cell_id, overlap_mask in zip(overlapping_cells, overlapping_cell_masks):
                        overlap_score, _ = compute_overlap(overlap_mask, cell_mask)

                        if overlap_score >= OVERLAP_THRESHOLD:
                            tmp_mask += overlap_mask

                            mask[np.where(mask == cell_id)] = 0
                            idx = [i for i, cell in enumerate(masks_stats[mask_id])
                                   if cell[0] == cell_id][0]
                            masks_stats[mask_id].pop(idx)

                # # If none overlapping do nothing

        if len(np.unique(tmp_mask)) > 1:
            cell_counter += 1
            final_mask[np.where(tmp_mask >= MIN_DETECTIONS)] = cell_counter

        cells_left = sum([len(stats) for stats in masks_stats])

    bin_mask = np.zeros(final_mask.shape)
    bin_mask[np.where(final_mask > 0)] = 255

    cv2.imwrite('results/final_bin.png', bin_mask)
    cv2.imwrite('results/final.png', final_mask)
    return final_mask

############################################################
#  Merge
############################################################

def merge_models(model_dirs):
    """
    Merge mask results given by the multiple models
    """
    filenames = os.listdir(model_dirs[0])

    submit_dir = "{:%Y%m%d_%H%M%S}_merge".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    print('\nMerging results...')
    for filename in tqdm(filenames):
        masks = [cv2.imread(os.path.join(model_dir, filename), 0) for model_dir in model_dirs]
        result = merge_masks(masks)
        cv2.imwrite(os.path.join(submit_dir, filename), result)
    print('\nResults were stored in', submit_dir)
