"""
Test a sample of adversarial examples generated with an attack over a model.

The results are saved in a CSV file containing the following columns:

    - surrogate_model: Name of the model for which the adversarial examples
    were generated. This value may be the one used to obtain the model from a 
    Python package (for example, timm).

    - algorithm: Name of the algorithm used to generate the adversarial
    examples for the surregate_model.

    - eps: Maximum allowed distance between the original sample and the
    adversarial exmaple according to a specific distance function 
    (currrently, only the unifrm distance is used).

    - niter: Number of iterations used to generate the adversarial exmaples
    via the algorithm. If the algorithm is not iterative, then the value is
    'nan'.

    - target_model: Name of the model that the adversarial examples are tested
    agains. This value may be the one used to obtain the model from a 
    Python package (for example, timm).


    - time: number of seconds that the execution of all the adversarial examples
    againt the target model.

    - clean_acc: accuracy of the target_model over the original samples of the
    adversarial examples used. Specifically, if n is the number of adversarial
    examples used, x'_i is the i-th adversarial example, x_i is the original
    sample of x'_i and y_i is the correct class of x_i, then 

        clean_acc = #{x_i for all i=1, ..., n | target_model(x_i) == y_i} / n,
    
    where # denotes the number of elements in a set and target_model(x_i) is the
    class predicted by the target_model for the sample x_i.

    - adv_acc: rate of adversarial examples incorrectly classified by the target
    model. The complement of this (1-value) is the attack success rate of
    the transferability attack. The original sample of the adversarial examples 
    used to obtain this metric must be correctly classified by the target model.

    - dir: relative path to the directory that stores the adversarial images.

    - nsamples: number of adversarial examples provided. This number also means
    the number of adversarial examples that fool the surrogate
    model but whose original sample is correctly classified by the surrogate_model
    (see note 2).

    - surrogate_model_acc: accuracy of the surrogate_model over the adversarial 
    examples generated for the original dataset (not provided in this program).

    - clean_nexamples: number of original samples of the adversarial examples 
    correctly classified by the target model.

    - surrogate_model_adv_acc: accuracy of the surrogate_model over the adversarial
    examples

Note 1: There may be confussion on how the adv_acc is obtained. Instead of using all the
    adversarial examples provided in the script input to test the target model, the
    adversarial examples used in the transferability attack must meet a condition:
    its original sample is correctly classified by the target model. If we used all
    the adversarial examples, whether or not is correctly classified by the target
    model, then the attack success rate may be increased not because the adversarial
    examples are "better" fooling the target model, but because the target model is
    intially not able to classify the original sample. Then, this can mask the
results of the transferability attack.

Note 2: The original dataset of samples is 5000. However, we consider 
    in this experiment only adversarial examples whose orignal sample is correctly
    classified by the surrogate_model and whose prediction is different from the 
    correct one.
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from argparse import ArgumentParser
import csv

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torchvision.datasets.folder import  default_loader
import numpy as np
from art.estimators.classification import PyTorchClassifier

from utils.model import get_model
from utils.data import get_preprocessing, LimitedImageNet
from utils.args import positive_number
from utils.constants import (
    CLIP_VALUES,
    DATASET_PATH_PARTS,
    IMAGES_DIR,
    INDICES_NAME,
    PARAM_ATTACKER,
    PARAM_FILENAME,
    PARAM_NITER,
    PARAM_PERT1,
    PARAM_PERT2,
    SUMMARY_FILENAME,
    SURROGATE_MODEL_NAME,
    NSAMPLES,
)

def path(filename):
    """Return an absolute path to a file in the current directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

logging.config.fileConfig(path("logging.conf"))
logger = logging.getLogger(__name__)

def get_argparse():
    
    parser = ArgumentParser()
    parser.add_argument(
        "--datadirs",
        nargs="+",
        type=Path,
        help=(
            ""
        )
    )
    parser.add_argument(
        "--rootoutputdir",
        type=Path,
        help="Output directory name (not path)",
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
        "--dataset-dir",
        default=Path(*DATASET_PATH_PARTS),
        help="Path to the ImageNet dataset."
    )
    parser.add_argument(
        "--nthreads",
        default=2, 
        type=positive_number, 
        help="Number of threads to be used."
    )
    parser.add_argument(
        "targetmodels",
        nargs="+",
        # choices=[
        #     "vgg16.tv_in1k",
        #     "inception_v3.tv_in1k",
        #     "resnet50.tv2_in1k",
        #     "inception_v3.tf_adv_in1k",
        #     "inception_resnet_v2.tf_ens_adv_in1k",
        #     "random-padding",
        #     "jpeg",
        #     "bit-reduction",
        #     "neural-representation-purifier",
        #     "vgg19.tv_in1k",
        #     "resnet152.tv2_in1k",
        #     "mobilenetv2_140.ra_in1k",
        # ],
        help=(
            ""
        )
    )
    return parser


if __name__ == '__main__':

    sys.path.append(os.path.abspath("./src"))
    parser = get_argparse()
    args = parser.parse_args()

    torch.set_num_threads(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    device = "cpu"
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "You specified the usage of the GPU, but no GPU is available."
            )
        device = "cuda"

    batch_size = args.batch_size
    
    for target_model_name in args.targetmodels:

        # Get those real images from ImageNet that have an adversary example
        logger.info(f"Reading the original ImageNet for the model '{target_model_name}'.")
        preprocessing = get_preprocessing(target_model_name)

        imagenet = LimitedImageNet(
            args.dataset_dir,
            "val",
            transform=preprocessing,
        )

        test_loader = data.DataLoader(
            imagenet,
            batch_size=len(imagenet),
            shuffle=False,
            num_workers=args.nthreads,
        )
        
        real_x, real_y = next(iter(test_loader))
        real_x = real_x.numpy()
        real_y = F.one_hot(real_y).numpy()

        # Load the target model
        logger.info(f"Loading the target model '{target_model_name}'")
        target_model = get_model(target_model_name)
        preprocesssing = get_preprocessing(target_model_name)
        target_model.eval()
        
        for datadir in tqdm(args.datadirs, desc=f"Transfering to '{target_model_name}'", unit="dirs"):

            # Check if the required dirs and files exist
            if not datadir.is_dir():
                logger.error(
                    f"'{datadir}' is not a valid directory."
                )
                continue

            param_file = Path(datadir, PARAM_FILENAME)

            if not param_file.exists():
                logger.error(
                    f"The file '{param_file}' is not present in '{datadir}'."
                )
                continue
            
            images_dir = Path(datadir, IMAGES_DIR)

            if not images_dir.is_dir():
                logger.error(
                    f"The directory '{images_dir}' is not a valid directory."
                )
                continue

            orig_summary_file = Path(datadir, SUMMARY_FILENAME)

            if not orig_summary_file.exists():
                logger.error(
                    f"The ifle '{orig_summary_file}' is not present in '{datadir}'."
                )
                continue
            
            # Check if the required parameters exist and get them
            with open(param_file, "r") as f:
                param_file_yaml = yaml.unsafe_load(f)

            if SURROGATE_MODEL_NAME not in param_file_yaml:
                logger.error(
                    f"The file '{param_file}' does not contain a valid surrogate" \
                    f" model field ('{SURROGATE_MODEL_NAME}')."
                )
                continue

            surrogate_model = param_file_yaml[SURROGATE_MODEL_NAME]

            if PARAM_PERT1 in param_file_yaml:
                eps = param_file_yaml[PARAM_PERT1]
            elif PARAM_PERT2 in param_file_yaml:
                eps = param_file_yaml[PARAM_PERT2]
            else:
                logger.error(
                    f"The file '{param_file}' does not contain a valid perturbation" \
                    f" field ('{PARAM_PERT1}' or '{PARAM_PERT2}')."
                )
                continue
            
            if PARAM_ATTACKER not in param_file_yaml:
                logger.error(
                    f"The file '{param_file}' does not contain a valid algorithm" \
                    f" field ('{PARAM_ATTACKER}')."
                )
                continue
            attacker_name = param_file_yaml[PARAM_ATTACKER]

            if PARAM_NITER not in param_file_yaml:
                logger.error(
                    f"The file '{param_file}' does not contain a valid iteration"
                    f" field ('{PARAM_NITER}')."
                )
                continue

            niter = param_file_yaml[PARAM_NITER]

            with open(orig_summary_file, "r") as f:
                reader = csv.DictReader(f)
                # It is expected that there are just two rows: the header
                # and the results during the generation of adversarial examples
                rows = list(reader)
                if len(rows) == 0:
                    logger.error(
                        f"The file '{SUMMARY_FILENAME}' in '{datadir}' does not contain any data."
                    )
                    continue

                clean_acc = float(rows[0]["clean_acc"])
                adv_acc = float(rows[0]["adv_acc"])

            
            surrogate_model_acc = clean_acc
            surrogate_model_adv_acc = adv_acc

            # Check if the result directory exists
            name_dir = f"{target_model_name}_{datadir.name}"
            outputdir = Path(args.rootoutputdir, name_dir)

            if outputdir.exists():
                logger.error(
                    f"The file '{outputdir}' already exists."
                )
                continue

            # Load the adversarial images
            adv_x = []
            adv_index = []
            adv_y = []
            
            logger.info(f"Reading adversarial images from '{images_dir}'")

            start_time = time.time()

            for class_dir in images_dir.iterdir():
                for image_file in class_dir.iterdir():
                    adv_y.append(int(class_dir.name))
                    adv_index.append(int(image_file.name.split(".")[0]))
                    image = preprocesssing(default_loader(image_file))
                    adv_x.append(image)

            adv_x = torch.stack(adv_x)
            adv_y = F.one_hot(torch.tensor(adv_y))
            adv_index = np.array(adv_index)
            nimages = len(adv_x)
            logger.info(f"{nimages} adversarial images were read")

            
            copy_real_x = real_x[adv_index].copy()
            copy_real_y = real_y[adv_index].copy()

            # As we are not training, the criterion is ignored
            criterion = nn.CrossEntropyLoss()
            classifier = PyTorchClassifier(
                target_model, 
                criterion,
                target_model.default_cfg['input_size'],
                imagenet.nb_classes,
                clip_values=CLIP_VALUES,
                device_type="gpu" if device == "cuda" else "cpu"
            )

            logger.info("Calculating clean accuracy (considering only the original images of the adversarial examples loaded)")
            # Predictions over the original dataset
            with torch.no_grad():
                clean_predictions = classifier.predict(copy_real_x, args.batch_size)

            # Set to true the indices of the correctly classified images
            correctly_pred = np.argmax(clean_predictions, axis=1) == np.argmax(copy_real_y, axis=1)
            clean_nexamples = np.sum(correctly_pred)
            torch.cuda.empty_cache()
            clean_accuracy = clean_nexamples / len(copy_real_y)
            logger.info(f"Accuracy over clean examples (considering only the original images of the adversarial examples loaded): {clean_accuracy:.4f}")
            
            # Get those adversarial images whose original image is correctly
            # classified by the target model
            orig_adv_index = adv_index.copy()
            adv_x = adv_x[correctly_pred]
            adv_y = adv_y[correctly_pred]
            adv_index = adv_index[correctly_pred]

            nadvs = len(adv_x)
            logger.info(f"Using {nadvs} images from the total {len(imagenet)}")
            
            logger.info("Calculating accuracy over adversarial examples")
            # Predictions over the original dataset
            with torch.no_grad():
                adv_predictions = classifier.predict(adv_x, args.batch_size)

            # Set to true the indices of the correctly classified images
            successful_transfer = np.argmax(adv_predictions, axis=1) == np.argmax(adv_y, axis=1)

            adv_accuracy = successful_transfer.sum().item() / len(adv_y)
            logger.info(f"Accuracy over adversarial examples: {adv_accuracy:.4f}")

            end_time = time.time()

            outputdir.mkdir()
            indices_path = Path(outputdir, INDICES_NAME)

            logger.info(f"Creating the indices file in the directory '{outputdir}'")
            with open(indices_path, "w") as file:
                yaml.dump(
                    {"good_prediction": orig_adv_index.tolist()}, 
                    file
                )
                yaml.dump(
                    {
                        "bad_prediction": orig_adv_index[np.logical_not(correctly_pred)].tolist()
                    },
                    file
                )
                yaml.dump({"sucessful_attack": adv_index[successful_transfer].tolist()}, file)
                yaml.dump({"failed_attack": adv_index[np.logical_not(successful_transfer)].tolist()}, file)

            rows = [
                [
                    "surrogate_model",
                    "algorithm",
                    "eps",
                    "niter",
                    "target_model",
                    "time",
                    "clean_acc",
                    "adv_acc",
                    "dir",
                    "nsamples",
                    "surrogate_model_acc",
                    "clean_nexamples",
                    "surrogate_model_adv_acc",
                ],
                [
                    surrogate_model,
                    attacker_name,
                    eps,
                    niter,
                    target_model_name,
                    end_time - start_time,
                    clean_accuracy,
                    adv_accuracy,
                    datadir,
                    nimages,
                    surrogate_model_acc,
                    clean_nexamples,
                    surrogate_model_adv_acc,
                ],
            ]

            logger.info(f"Creating the summary in '{outputdir}'")
            with open(Path(outputdir, SUMMARY_FILENAME), "w") as file:
                writer = csv.writer(file)
                writer.writerows(rows)
