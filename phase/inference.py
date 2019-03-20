from phase import *

from tqdm import tqdm
from merge import merge_models

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/")

############################################################
#  Detection
############################################################

def detect(model, dataset_dir, single_dir=False):
    """Run detection on images in the given directory."""

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        print('Creating results directory:', RESULTS_DIR)
        os.makedirs(RESULTS_DIR)
    submit_dir = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = PhaseDataset()
    dataset.load_phase(dataset_dir, single_dir)
    dataset.prepare()

    # Load over images
    submission = []
    for image_id in tqdm(dataset.image_ids):
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        source_id = dataset.image_info[image_id]["id"]

        out_path = os.path.join(submit_dir, '%s.png' % str(source_id))

        mask = np.argmax(r['masks'], 2)
        cv2.imwrite(out_path, mask)

    return submit_dir


############################################################
#  Command Line
############################################################

def run_inference(model, weights, infer_data, single_dir):
    # Load weights
    print("\nLoading weights:", weights)
    model.load_weights(weights, by_name=True)

    # Run inference
    print('\nRunning inference...')
    return detect(model, infer_data, single_dir)

def is_valid_weights(weights):
    return os.path.splitext(weights)[1] == '.h5' and not weights.startswith('.')

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Inference for phase contrast images')
    parser.add_argument('--infer_data', required=True,
                        metavar="/path/to/infer/dataset/",
                        help='Directory of the inference data')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights",
                        help="Path to weights")
    parser.add_argument('--single_dir', required=False,
                        action='store_true',
                        default=False,
                        help='Data is in a single directory (default=false)')
    args = parser.parse_args()

    # Select inference data directory
    args.infer_data = os.path.join(ROOT_DIR, args.infer_data)
    if not os.path.isdir(args.infer_data):
        print(args.infer_data, 'is not a directory.')
        exit()
    print("Inference dataset: ", args.infer_data)

    if args.single_dir:
        print('Infering on data located in a single directory.')

    # Select weights file to load
    args.weights = os.path.join(ROOT_DIR, args.weights)
    if not os.path.exists(args.weights):
        print(args.weights, 'does not exist.')
        exit()

    # Select log directory
    if not os.path.isdir(DEFAULT_LOGS_DIR):
        print('Creating log directory:', DEFAULT_LOGS_DIR)
        os.mkdir(DEFAULT_LOGS_DIR)
    print("Logs: ", DEFAULT_LOGS_DIR, "\n")

    # Configuration
    config = PhaseInferenceConfig()
    #config.display()

    # Create model
    print('Creating model...')
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Single weights file
    if os.path.isfile(args.weights) and is_valid_weights(args.weights):
        run_inference(model, args.weights, args.infer_data, args.single_dir)

    # Multiple weights files located in the given directory
    elif os.path.isdir(args.weights):
        weights_files = list(filter(is_valid_weights, os.listdir(args.weights)))
        results = list()
        for it, weight_file in enumerate(weights_files):
            print('\nModel {}/{}:'.format(it+1, len(weights_files)))
            weight_file = os.path.join(args.weights, weight_file)
            results.append(run_inference(model, weight_file,
                args.infer_data, args.single_dir))
        merge_models(results)


    else: print('\nInvalid weights file')
