

DATASET_PATH_PARTS = ["data", "imagenet"]

# Name of the file that stores the parameters about the modified images
PARAM_FILENAME = "run.yaml"

# Choices for the name of the perturbation parameter in ``PARAM_FILENAME``.
PARAM_PERT1 = "eps"
PARAM_PERT2 = "epsilon"

# Name of the parameter in `PARAM_FILENAME`` that stores the name of the model
# attacked by generating the modified image.
SURROGATE_MODEL_NAME = "model_name"

# Name of the parameter in ``PARAM_FILENAME`` that stored the algorithm
# used to generate the modified images.
PARAM_ATTACKER = "attacker_name"

PARAM_NITER = "max_iter"

# Name of the directory that stores the modified images.
IMAGES_DIR = "adversarial_images"

# Minimum and maximum values of the data
CLIP_VALUES = (0.0, 1.0)

# Name of the file where the indices of the images in the used dataset
# whose attack failed; i.e. its adversarial example prediction was the 
# same as the original image.
INDICES_NAME = "indices.yaml"

SUMMARY_FILENAME = "summary.csv"

# Number of samples used during the generation of adversarial examples
# Currently, this is limited to the one set in Robustbench
NSAMPLES = 5000