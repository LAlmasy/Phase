import numpy as np
import skimage.morphology

############################################################
#  Combine neighbour labels
############################################################

'''
Transform the list of binary sets containing the merged neighbours into
a regrouped list of merged neighbours.
e.g. [{2,3}, {3,8}, {4,12}] --> [{2,3,8}, {4,12}]
This is done using recursivity until no fusions of set are possible.
'''
def combine_set(to_merge):
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
        return combine_set(to_merge)
    else:
        return to_merge

'''
Merge neighbour cells. A dilaion is applied to each cell's mask and the result
is compared with the masks of the other cells. If the intersection is big
enough the cells are considered to be the same and are merged.

Returns: the updated mask with merged cells.
'''
def merge_neighbours(mask):
    ids = np.unique(mask)[1:]
    if len(ids) == 0: return mask
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

    to_merge = combine_set(to_merge)

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
