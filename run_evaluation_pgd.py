import sys
import os
import argparse
import logging
import logging.config
import random
import time
import yaml
import csv
from pathlib import Path

import torchvision
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.evasion import (
    ProjectedGradientDescent,
    AutoProjectedGradientDescent,
    AutoConjugateGradient,
    RescalingAutoConjugateGradient,
)
from art.estimators.classification import PyTorchClassifier
import numpy as np

sys.path.append(os.path.abspath("./src"))

from utils.data import get_preprocessing, LimitedImageNet
from utils.args import positive_number
from utils.model import get_model
from utils.io import read_yaml, overwrite_config
from utils.configuration import get_configurations

def path(filename):
    """Return an absolute path to a file in the current directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

logging.config.fileConfig(path("logging.conf"))
logger = logging.getLogger(__name__)

DATASET_PATH_PARTS = ["data", "imagenet"]

# Minimum and maximum values of the data
CLIP_VALUES = (0.0, 1.0)

# Name of the file where the execution parameters are stored
EX_PARAMS_FILE = "run.yaml"

# Name of the file where the indices of the images in the used dataset
# whose attack failed; i.e. its adversarial example prediction was the 
# same as the original image.
INDICES_NAME = "indices.yaml"

# Name of the directory where the adversarial images are stored
ADV_DIR_NAME = "adversarial_images"

SUMMARY_FILENAME = "summary.csv"

def argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cmd-param",
        type=str,
        nargs="*",
        help='list of "param:cast type:value"'
        + " ex)  model_name:str:XXX solver.params.num_sample:int:10",
    )
    parser.add_argument(
        "--gpu", 
        action="store_true",
        help=(
            "Indicates if the GPU is used to process the operations.\n" \
            "If no CUDA GPU is available, then the program exits."
        ),
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=positive_number,
        help="Number of images processed simultaneously.",
    )
    parser.add_argument(
        "--nthreads", 
        default=2,
        type=positive_number,
        help="Nuumber of threads to be used."
    )
    parser.add_argument(
        "--dataset-dir",
        default=Path(*DATASET_PATH_PARTS),
        help="Path to the ImageNet dataset."
    )
    # Indicar que solo para los modelos que no se encuentan en timm
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to the directory that contains the models files."
    )
    parser.add_argument(
        "--save-adversarial",
        action="store_true",
        help="whether to save the adversarial images in a folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="whether the output directory is overwritten if it exists."
    )
    parser.add_argument(
        "param_file", 
        type=Path,
        help=(
            "YAML file that contains the configuration of " \
            "the scenarios."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory name (not path). It must not exist.",
    )
    return parser

def reproducibility():
    """
    Set the ennvironment variables and configurations so subsequent runnings
    of the experiments can be reproduced.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    
    parser = argparser()
    args = parser.parse_args()
    torch.set_num_threads(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    args.output_dir.mkdir(exist_ok=args.overwrite)

    device = "cpu"
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You specified the usage of the GPU, but no GPU is available."
            )
        device = "cuda"

    config = read_yaml(args.param_file)
    
    # overwrite by cmd_param
    if args.cmd_param:
        config = overwrite_config(args.cmd_param, config)

    logger.info("Loading the model")
    model = get_model(config.model_name)
    model.eval()
    transforms = get_preprocessing(config.model_name)
    
    logger.info("Loading the dataset")
    imagenet = LimitedImageNet(
        args.dataset_dir,
        "val",
        transform=transforms
    )

    loader = data.DataLoader(
        imagenet,
        batch_size=len(imagenet),
        num_workers=args.nthreads
    )

    x, y = next(iter(loader))
    x = x.numpy()
    y = F.one_hot(y).numpy()

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model, 
        criterion,
        model.default_cfg['input_size'],
        imagenet.nb_classes,
        clip_values=CLIP_VALUES,
        device_type="gpu" if device == "cuda" else "cpu"
    )

    logger.info("Calculating clean accuracy")
    # Predictions over the original dataset
    with torch.no_grad():
        clean_predictions = classifier.predict(x, config.batch_size)

    # Set to true the indices of the correctly classified images
    correctly_pred = np.argmax(clean_predictions, axis=1) == np.argmax(y, axis=1)

    torch.cuda.empty_cache()
    clean_accuracy = np.sum(correctly_pred) / len(y)
    logger.info(f"Accuracy over clean examples: {clean_accuracy:.4f}")

    if config.attacker_name == "PGD":
        algorithm = ProjectedGradientDescent(
        estimator=classifier,
        eps=config.eps,
        max_iter=config.max_iter,
        batch_size=config.batch_size
    )
    elif config.attacker_name == "APGD":
        algorithm = AutoProjectedGradientDescent(
        estimator=classifier,
        eps=config.eps,
        max_iter=config.max_iter,
        batch_size=config.batch_size,
        nb_random_init=1
    )
    elif config.attacker_name == "ACG":
        algorithm = AutoConjugateGradient(
        estimator=classifier,
        eps=config.eps,
        max_iter=config.max_iter,
        batch_size=config.batch_size,
        nb_random_init=1
    )
    elif config.attacker_name == "ReACG":
        algorithm = RescalingAutoConjugateGradient(
        estimator=classifier,
        eps=config.eps,
        max_iter=config.max_iter,
        batch_size=config.batch_size,
        nb_random_init=1
    )
        
    logger.info("Generating adversarial examples")
    start_time = time.time()
    x_adv = algorithm.generate(x, y)
    end_time = time.time()
    torch.cuda.empty_cache()

    logger.info("Calculating accuracy over adversarial examples")
    with torch.no_grad():
        adv_predictions = classifier.predict(x_adv, config.batch_size)

    torch.cuda.empty_cache()
    adv_accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    logger.info(f"Accuracy over adversarial examples: {adv_accuracy:.4f}")

    # Set to true the indices of the adversarial images not correctly
    # classified by the model
    adversarial_inds2 = np.argmax(adv_predictions, axis=1) != np.argmax(clean_predictions, axis=1)

    # Set to true the indices of the originally correctly classified images
    # but successfully attacked.
    adversarial_inds = np.logical_and(correctly_pred, adversarial_inds2)

    ex_params_path = Path(args.output_dir, EX_PARAMS_FILE)

    logger.info("Writing config into file")
    with open(ex_params_path, "w") as file:
        yaml.dump(dict(config), file)

    indices = np.array(range(x.shape[0]))

    # Get the index of the adversary examples classified
    # correctly, i.e., whose attack failed
    indices_path = Path(args.output_dir, INDICES_NAME)

    logger.info("Creating the indices file")
    with open(indices_path, "w") as file:
        yaml.dump({"good_prediction": indices[correctly_pred].tolist()}, file)
        yaml.dump({"bad_prediction": indices[np.logical_not(correctly_pred)].tolist()}, file)
        yaml.dump({"sucessful_attack": indices[adversarial_inds].tolist()}, file)
        yaml.dump({"failed_attack": indices[np.logical_and(correctly_pred, np.logical_not(adversarial_inds))].tolist()}, file)

    rows = [
        [
            "model",
            "algorithm",
            "threat_model",
            "criterion_name",
            "niter",
            "eps",
            "time",
            "clean_acc",
            "adv_accuracy",
        ],
        [
            config.model_name,
            config.attacker_name,
            config.threat_model,
            config.criterion_name,
            config.max_iter,
            config.eps,
            end_time - start_time,
            clean_accuracy,
            adv_accuracy,
        ],
    ]

    logger.info("Creating the summary")
    with open(Path(args.output_dir, SUMMARY_FILENAME), "w") as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    if args.save_adversarial:
        adv_images_dir = Path(args.output_dir, ADV_DIR_NAME)
        adv_images_dir.mkdir()
        
        logger.info(f"Saving generated adversarial images in '{adv_images_dir.resolve()}'")
        for index in indices[adversarial_inds]:
                
            class_dir = Path(adv_images_dir, str(np.argmax(y[index]).item()))
            class_dir.mkdir(exist_ok=True)
            image_cpu = torch.from_numpy(x_adv[index])
            image_name = f"{index}.png"
            torchvision.utils.save_image(
                image_cpu, Path(class_dir, image_name)
            )
