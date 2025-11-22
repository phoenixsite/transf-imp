import sys
import os
import argparse
import logging
import logging.config
import random
import datetime
import time
import yaml
from pathlib import Path

import torchvision
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import numpy as np

from utils.data import get_preprocessing, LimitedImageNet

from utils import read_yaml, overwrite_config, get_configurations

def path(filename):
    """Return an absolute path to a file in the current directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

logging.config.fileConfig(path("logging.conf"))
logger = logging.getLogger(__name__)

DATASET_PATH_PARTS = ["data", "imagenet"]


def argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory name (not path)",
        required=True,
        type=str,
    )
    parser.add_argument("-p", "--param", required=True, nargs="*", type=str)
    parser.add_argument(
        "--cmd_param",
        type=str,
        default=None,
        nargs="*",
        help='list of "param:cast type:value"'
        + " ex)  model_name:str:XXX solver.params.num_sample:int:10",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None)
    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument(
        "--image_indices",
        type=str,
        default=None,
        help="path to yaml file which contains target image indices",
    )
    parser.add_argument(
        "--export_level", type=int, default=60, choices=[10, 20, 30, 40, 50, 60]
    )
    parser.add_argument(
        "--dataset-dir",
        default=Path(*DATASET_PATH_PARTS),
        help="Path to the ImageNet dataset."
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="attack all images when this flag is on",
    )
    parser.add_argument(
        "-sa",
        "--save-adversarial",
        action="store_true",
        help="whether to save the adversarial images in a folder",
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
    
    sys.path.append(os.path.abspath("./src"))
    reproducibility()
    parser = argparser()
    args = parser.parse_args()
    logger = logger.setLevel(args.log_level)
    torch.set_num_threads(args.nthreads)
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)

    config = read_yaml(args.param)
    
    # overwrite by cmd_param
    if args.cmd_param:
        config = overwrite_config(args.cmd_param, config)
    
    config.update(get_configurations(args))

    model = robustbench.load_model(
        config.model_name,
        model_dir=os.path.join("..", "models"),
        dataset=config.dataset,
        threat_model=config.threat_model,
    )

    transforms = get_preprocessing(config.model_name)
    data_dir = os.path.join("..", "data")
    
    imagenet = LimitedImageNet(
        args.dataset_dir,
        "val",
        transforms=transforms
    )

    loader = data.DataLoader(
        imagenet,
        batch_size=len(imagenet),
        num_workers=args.nthreads
    )

    x, y = load_imagenet(
        n_examples=config.n_examples,
        data_dir=os.path.join(data_dir, "imagenet"),
        transforms_test=transforms,
    )
    x = x.numpy()
    y = F.one_hot(y).numpy()

    criterion = nn.CrossEntropyLoss()
    classifier = PyTorchClassifier(
        model, 
        criterion,
        model.default_cfg['input_size'],
        1000
    )

    clean_predictions = classifier.predict(x, config.batch_size)
    clean_accuracy = np.sum(np.argmax(clean_predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    logger.info(f"clean acc: {clean_accuracy:.2f}")

    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=config.eps,
        max_iter=config.max_iter,
        batch_size=config.batch_size
    )
    stime = time.time()
    x_adv = attack.generate(x, y)

    adv_predictions = classifier.predict(x_adv, config.batch_size)
    adv_accuracy = np.sum(np.argmax(adv_predictions, axis=1) == np.argmax(y, axis=1)) / len(y)
    logger.info("adversarial test examples: {}%".format(adv_accuracy * 100))
    adversarial_inds1 = np.argmax(clean_predictions, axis=1) == np.argmax(y, axis=1)
    adversarial_inds2 = np.argmax(adv_predictions, axis=1) != np.argmax(clean_predictions, axis=1)
    adversarial_inds = np.logical_and(adversarial_inds1, adversarial_inds2)

    if args.save_adversarial:
        today = datetime.date.today().isoformat()
        _time = ":".join(datetime.datetime.now().time().isoformat().split(":")[:2])
        output_dir = os.path.join(config.output_dir, today, _time)
        os.makedirs(output_dir, exist_ok=True)
        output_root_dir = os.path.join(
            output_dir,
            config.threat_model,
            config.dataset,
            config.model_name,
            "PGD",
        )
        os.makedirs(output_root_dir, exist_ok=True)
        output_images_dir = os.path.join(output_root_dir, "adversarial_examples")
        os.makedirs(output_images_dir, exist_ok=True)
        logger.info(f"Saving generated adversarial images in {output_images_dir}.")
        for index, is_adv in enumerate(adversarial_inds):
                is_adv = is_adv.item()
                
                if is_adv:
                    output_dir = os.path.join(output_images_dir, str(np.argmax(y[index]).item()))
                    os.makedirs(output_dir, exist_ok=True)
                    image_cpu = torch.from_numpy(x_adv[index])
                    image_name = f"{index}.png"
                    torchvision.utils.save_image(
                        image_cpu, os.path.join(output_dir, image_name)
                    )

    run_yaml_path = os.path.join(
        output_root_dir,
        "run.yaml",
    )
    if not os.path.exists(run_yaml_path):
        with open(run_yaml_path, "w") as file:
            yaml.dump(dict(config), file)

    device = (
        torch.device(f"cuda:{config.gpu_id}")
        if torch.cuda.is_available() and config.gpu_id is not None
        else torch.device("cpu")
    )
    #_robust_acc, _, _ = compute_accuracy(x_adv, y, config.batch_size, model, device)
    _robust_acc = adv_accuracy
    # Get the index of the adversary examples classified
    # correctly, i.e., whose attack failed
    failed_indices_path = os.path.join(
        output_root_dir,
        "failed_indices.yaml",
    )
    if not os.path.exists(failed_indices_path):
        with open(failed_indices_path, "w") as file:
            yaml.dump({"indices": np.where(_robust_acc)[0].tolist()}, file)

    robust_acc = 100 * (_robust_acc.sum() / config.n_examples)
    attack_success_rate = 100 - robust_acc
    short_summary_path = os.path.join(output_root_dir, "short_summary.txt")
    
    msg = f"\ntotal time (sec) = {time.time() - stime:.3f}\nclean acc(%) = {clean_accuracy:.2f}\nrobust acc(%) = {robust_acc:.2f}\nASR(%) = {attack_success_rate:.2f}"
    
    with open(short_summary_path, "w") as f:
        f.write(msg)
    logger.info(msg)