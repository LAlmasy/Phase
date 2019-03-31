# MA1 Project - Convolutional deep neural network for cell segmentation

This project uses a conculutional deep neural networks to segment phase contrast images.

- [mrcnn](./mrcnn): this directory contains the Mask-RCNN implementation of Matterport,
available at: https://github.com/matterport/Mask_RCNN,
which is used as a basis for the neural network.

- [phase](./phase): this directory contains the custom configurations of the neural network and the scripts to make predictions
and process the resulting masks.

- [weights](./weights): use this directory to store the weights of the neural network, trained weights can be downloaded at:
https://drive.google.com/drive/folders/1D--_sGyH7pUPGUmQ6D2EVig2j2tbrJsf?usp=sharing.

- [results](./results): predicted masks will be stored in this directory.

The [report](./Report.pdf) of the project conatains details on the aim of the project, the implementation, and the analysis of its results and performances.
