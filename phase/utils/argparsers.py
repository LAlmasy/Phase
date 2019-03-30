from argparse import ArgumentParser

def parse_inference_args():
    parser = ArgumentParser(description='Inference for phase contrast images')
    parser.add_argument('--infer_data', required=True,
                        metavar="/path/to/infer/dataset/",
                        help='Directory of the inference data')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights/",
                        help="Path to weights")
    parser.add_argument('--not_single_dir', required=False,
                        action='store_true',
                        default=False,
                        help='Data is not in a single directory (default=false)')
    parser.add_argument('--merge', required=False,
                        action='store_true',
                        default=False,
                        help='Directly merge resulting masks (default=false)')
    return parser.parse_args()

def parse_post_processing_args():
    parser = ArgumentParser(description='Mask and sequence post-processing')
    parser.add_argument('--dir', required=True,
                        metavar="/path/to/masks/",
                        help='Path masks')
    parser.add_argument('--merge', required=False,
                        action='store_true',
                        default=False,
                        help='Merge neighbouring labels (default=false)')
    parser.add_argument('--track', required=False,
                        action='store_true',
                        default=False,
                        help='Track cells in the sequence (default=false)')
    return parser.parse_args()

def parse_view_arguments():
    parser = ArgumentParser(description='Visualisation of an image and its mask')
    parser.add_argument('--img', required=True,
                        metavar="/path/to/image.*",
                        help='Path to image')
    parser.add_argument('--mask', required=True,
                        metavar="/path/to/mask.*",
                        help="Path to mask")
    parser.add_argument('--borders', required=False,
                        action='store_true',
                        default=False,
                        help='View the borders (default=false)')
    return parser.parse_args()
