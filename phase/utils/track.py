import numpy as np

############################################################
#  Track cells between frames
############################################################

'''
Create a new mask based on the input mask, on which the cells that were
on the previous mask keep the same label value.
'''
def make_mask(mask, prev_mask):
    # List of cell ids in the mask
    cell_ids = list(np.unique(mask)[1:])
    # List of available grey value
    available = list(range(1, cell_ids[-1]+1))
    # Look up table to convert the mask to its new values
    lut = np.zeros(cell_ids[-1]+1)

    # For each cell in the previous mask
    for prev_cell_id in np.unique(prev_mask)[1:]:
        if prev_cell_id not in available: break
        max_overlap = 0
        cell = np.zeros(mask.shape)
        best_id = 0
        # Find the best match between the mask's cells
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
