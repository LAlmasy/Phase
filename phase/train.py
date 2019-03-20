from phase import *

from imgaug import augmenters as iaa

############################################################
#  Training
############################################################

def train(model, train_dir, val_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = PhaseDataset()
    dataset_train.load_phase(train_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PhaseDataset()
    dataset_val.load_phase(val_dir)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    config = PhaseConfig()

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                augmentation=augmentation,
                layers='all')

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument('--train', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--val', required=True,
                        metavar="Validation directory",
                        help="Directory to run validation on")
    args = parser.parse_args()

    args.train = os.path.join(ROOT_DIR, args.train)
    args.val = os.path.join(ROOT_DIR, args.val)

    print("Training data: ", args.train)
    print("Validation data: ", args.val)

    # Select log directory
    if not os.path.isdir(DEFAULT_LOGS_DIR):
        print('Creating log directory:', DEFAULT_LOGS_DIR)
        os.mkdir(DEFAULT_LOGS_DIR)
    print("Logs: ", DEFAULT_LOGS_DIR, "\n")

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = os.path.join(ROOT_DIR, args.weights)
		if not os.path.isfile(weights_path) or os.path.splitext(weights_path)[1] != '.h5':
			print('Invalid input weight file')
			exit()

    # Configuration
    config = PhaseConfig()
    #config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    print("\nLoading weights ", weights_path)
    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    #Run training
    print('\nTraining...')
    train(model, args.train, args.val)
