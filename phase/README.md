# Usage

**All the scripts should be executed from the phase directory (this directory) as they use relative paths.**

## Making predictions

To make prediction, use [inference.py](./inference.py).

The required arguments are:
- `--infer_data=<path/to/data/>` - The path to the raw data to make mask predictions on. This path is relative to the root of the project.
- `--weights=<path/to/weights>` - The path to the weight file(s) for the CNN to use. This path is relative to the root of the project.
It may be the path to a single `.h5` weight file or to a directory containing multiple weight files.

The options are:
- `--merge` - If multiple weight files are used, directly merge the resulting masks. If this option is not used the results will not
be directly merged, but can be merged using [merge.py](./merge.py)
- `--not_single_dir` - The data directory contains sub-directories that each contain a single raw image named `raw.tif`.
If this option is not used the data directory directly contains the raw images.

### Examples:

- To make predictions using a single weight file:

`python inference.py --infer_data=data/ --weights=weights/Usiigaci_1.h5`

- To make predictions using multiple weight files:

`python inference.py --infer_data=data/ --weights=weights/`

- To make predictions using multiple weight files and merge the resulting masks:

`python inference.py --infer_data=data/ --weights=weights/ --merge`


## Merging masks from different networks

To merge masks, use [merge.py](./merge.py). The arguments are the absolute paths to the directories containing the masks.
The appropriate command is give at the end of the prediction.

### Example:

`python merge.py /Users/.../Phase/results/net1/ /Users/.../Phase/results/net2/`

## Post-processing

To make post-processing, use [postprocessing.py](./postprocessing.py).

The required argument is:
- `--dir=<path/to/masks/>` - Path to the directory containing the masks to process. This path is relative to the [results](../results) folder.

The options are:
- `--merge` - Merge neighbouring/divided cells.
- `--track` - Track cells between frames to assign them the same value.

At least one option has to be selected.

### Example:
To merge neighbouring/divided cells and track cells in a sequence:

`python postprocessing.py --dir=exp1/ --merge --track`

## Visualisation

### Compare original image with mask
To make a visual comparison between the original raw image and the mask, use:

`python view.py --img=data/exp0012.jpg --mask=results/exp0012.png`

Adding the `--borders` options changes the visualisation from a superposition of the two images to a display of the mask borders on the image.

### Make a video
To make a video of the sequence of masks, use:

`python video.py results/seqence1/`

The argument being the path to the mask images.

