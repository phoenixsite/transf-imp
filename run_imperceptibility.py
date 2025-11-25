"""
Calculate imperceptibility metrics of ImageNet images.
"""

import sys
import argparse
import logging
import logging.config
import yaml
import csv
import os
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
from torch.utils import data
from torchvision.datasets.folder import  default_loader
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from utils.data import get_preprocessing, LimitedImageNet, INITIAL_TRANSFORM
from utils.args import positive_number

def path(filename):
    """Return an absolute path to a file in the current directory."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

logging.config.fileConfig(path("logging.conf"))
logger = logging.getLogger(__name__)

DATASET_PATH_PARTS = ["data", "imagenet"]

# Name of the file that stores the parameters about the modified images
PARAM_FILENAME = "run.yaml"

# Choices for the name of the perturbation parameter in ``PARAM_FILENAME``.
PARAM_PERT1 = "eps"
PARAM_PERT2 = "epsilon"

# Name of the parameter in `PARAM_FILENAME`` that stores the name of the model
# attacked by generating the modified image.
PARAM_MODEL = "model_name"

# Name of the parameter in ``PARAM_FILENAME`` that stored the algorithm
# used to generate the modified images.
PARAM_ATTACKER = "attacker_name"

PARAM_NITER = "max_iter"

# Name of the directory that stores the modified images.
IMAGES_DIR = "adversarial_images"

# Similarity metric names
SSIM = "ssim"
LPIPS = "lpips"
PSNR = "psnr"
FID = "fid"

# Object interfaces of the imperceptability metrics 
SSIM_METRIC = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
PSNR_METRIC = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3))
LPIPS_ALEX = LearnedPerceptualImagePatchSimilarity(
    net_type="alex", reduction="none", normalize=True
)
LPIPS_VGG = LearnedPerceptualImagePatchSimilarity(
    net_type="vgg", reduction="none", normalize=True
)
LPIPS_SQUEEZE = LearnedPerceptualImagePatchSimilarity(
    net_type="squeeze", reduction="none", normalize=True
)

# Header of the resulting CSV file
CSV_ROWS = [
    "model",
    "niter",
    "epsilon",
    "algorithm",
    "ssim_mean",
    "ssim_std",
    "psnr_mean",
    "psnr_std",
    "lpips_alex_mean",
    "lpips_alex_std",
    "lpips_vgg_mean",
    "lpips_vgg_std",
    "lpips_squeeze_mean",
    "lpips_squeeze_std",
    "dir",
]

# Number of images in the selected subset of ImageNet
NEXAMPLES = 5000

def get_argparse():
    """
    Set the arguments and options of the script.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Calculate similarity metrics of ImageNet modified images." \
            "The metrics are SSIM\n" \
            "(Structural Similarity Index Measure), 'PSNR (Peak\n" \
            "Signal-to-Noise Ratio), LPIPS (Learned Perceptual Image Path\n" \
            "Similarity) and FIPS (Frèchet Inception Distance)."
        ),
        formatter_class=argparse.RawTextHelpFormatter
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
        "--append",
        action="store_true",
        help=(
            "If the output file exists and this options is set, then" \
            " its contents are appended to the results."
        )
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help=(
            "CSV file where the results are written." \
            " Its columns are the following:\n"
            "\t1. Model name\n" \
            "\t2. Number of iterations\n" \
            "\t3. Maximum perturbation\n" \
            "\t4. Algorithm\n" \
            "\t5. Mean of SSIM \n" \
            "\t6. Standard deviation of SSIM \n" \
            "\t7. Mean of PSNR \n" \
            "\t8. Standard deviation of PSNR \n" \
            "\t9. Mean of LPIPS (Alex)\n" \
            "\t10. Standard deviation of LPIPS (Alex)\n" \
            "\t11. Mean of LPIPS (Alex)\n" \
            "\t12. Standard deviation of LPIPS (VGG)\n" \
            "\t13. Mean of LPIPS (Squeeze)\n" \
            "\t14. Standard deviation of LPIPS (Squeeze)\n" \
            "\t15. Directory used\n" \
            "It is necessary to specify the metric with the option --metric\n" \
            "in order to calcualte its mean and standard deviation. If not,\n" \
            "a hyphen ('-') written."
        ),
    )
    parser.add_argument(
        "testimagesdirs",
        nargs="+",
        type=Path,
        help=(
            "List of directory paths where the modified images are stored.\n" \
            "The filesystem structure of each directory must follow the following structure:\n" \
            "test_image_dir/\n" \
            "\trun.yaml\n" \
            "\tadversarial_images/\n" \
            "\t\t0/\n" \
            "\t\t\t1127.png\n" \
            "\t\t\t2927.png\n" \
            "\t\t...\n" \
            "\t\t927/\n" \
            "\t\t\t1764.png\n" \
            "\t\t\t2362.png\n" \
            "so the name of the first path corresponds to the\n" \
            "index of the label assigned to the image. The name\n" \
            "of the image files corresponds to the index of the\n" \
            "image in the subset of ImageNet used in Robustbench.\n" \
            "Moreover, each directory must contain a file named\n" \
            "'run.yaml' with the following keys:\n" \
            "\t - model_name\n" \
            "\t - epsilon (or eps)\n" \
            "\t - attacker_name\n" \
            "\t - niter\n" \
            "The value of 'model_name' must be a model available in the package" \
            " timm."
        ),
    )
    return parser


if __name__ == "__main__":
    
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

    already_read_dirs = []
    rows = []

    if args.output_file.exists():
        if not args.append:
            raise ValueError(
                f"The file '{args.output_file}' exists. Specifiy one that do not exists."
            )
        else:
            with open(args.output_file, "r") as file:
                reader = csv.reader(file)
                
                for row in reader:
                    already_read_dirs.append(row[-1])
                    rows.append(row)
    else:
        rows.append(CSV_ROWS)

    metrics = []
    metrics.append((SSIM, SSIM_METRIC.to(device)))
    metrics.append((PSNR, PSNR_METRIC.to(device)))
    metrics.append((f"{LPIPS}_alex", LPIPS_ALEX.to(device)))
    metrics.append((f"{LPIPS}_vggg", LPIPS_VGG.to(device)))
    metrics.append((f"{LPIPS}_squeeze", LPIPS_SQUEEZE.to(device)))

    full_original_images = {}

    # First, the Robustbench ImageNet dataset is loaded. A transformed copy
    # of the dataset is stored in memory. Mantaining these copies speeds up
    # the process. Careful: this may need a lot memory
    for testimagesdir in tqdm(
        args.testimagesdirs, desc="Preprocessing datasets", unit="directories"
    ):
        error = False

        if not testimagesdir.is_dir():
            logger.error(f"'{testimagesdir}' is not a valid directory.")
            error = True

        run_file = Path(testimagesdir, PARAM_FILENAME)

        if not run_file.exists():
            logger.error(f"The file '{run_file}' is not present in '{testimagesdir}'.")
            error = True

        images_dir = Path(testimagesdir, IMAGES_DIR)

        if not images_dir.exists():
            logger.error(f"The directory '{IMAGES_DIR}' is not in '{testimagesdir}'.")
            error = True

        # The loop break at the end so the user is informed
        # of all the initial problems.
        if error:
            continue

        with open(run_file, "r") as f:
            run_file_yaml = yaml.unsafe_load(f)

        model_name = run_file_yaml[PARAM_MODEL]

        if not model_name in full_original_images:
            logger.debug(f"Reading the dataset for the model '{model_name}'.")
            preprocessing = get_preprocessing(model_name)

            original_imagenet = LimitedImageNet(
                args.dataset_dir,
                "val",
                transform=preprocessing,
            )

            test_loader = data.DataLoader(
                original_imagenet,
                batch_size=len(original_imagenet),
                shuffle=False,
                num_workers=args.nthreads,
            )
            
            original_images, _ = next(iter(test_loader))

        if model_name not in full_original_images:
            full_original_images[model_name] = ([testimagesdir], original_images)
        else:
            full_original_images[model_name][0].append(testimagesdir)

    for model_name in full_original_images:
        for testimagesdir in tqdm(
            full_original_images[model_name][0],
            desc=f"Using data preprocessed images from {model_name}",
            unit="directories",
        ):
            
            if str(testimagesdir) in already_read_dirs:
                continue

            original_images = full_original_images[model_name][1]

            run_file = Path(testimagesdir, PARAM_FILENAME)

            with open(run_file, "r") as f:
                run_file_yaml = yaml.unsafe_load(f)

            if PARAM_PERT1 in run_file_yaml:
                eps = run_file_yaml[PARAM_PERT1]
            elif PARAM_PERT2 in run_file_yaml:
                eps = run_file_yaml[PARAM_PERT2]
            else:
                logger.error(
                    f"The file '{run_file}' does not contain a valid perturbation"
                    f" field ('{PARAM_PERT1}' or '{PARAM_PERT2}')."
                )
                continue

            attacker_name = "UNK"
            if PARAM_ATTACKER in run_file_yaml:
                attacker_name = run_file_yaml[PARAM_ATTACKER]
            else:
                attacker_name = "PGD"

            if not PARAM_NITER in run_file_yaml:
                logger.error(
                    f"The file '{run_file}' does not contain a valid iteration"
                    f" field ('{PARAM_NITER}')."
                )
                continue

            niter = run_file_yaml[PARAM_NITER]
            images_dir = Path(testimagesdir, IMAGES_DIR)
            modified_images = []
            index_test = []

            logger.debug("Reading test images.")

            for class_dir in images_dir.iterdir():
                for image_file in class_dir.iterdir():
                    index_test.append(int(image_file.name.split(".")[0]))
                    image = INITIAL_TRANSFORM(default_loader(image_file))
                    modified_images.append(image)

            modified_images = torch.stack(modified_images)
            nimages = len(modified_images)
            logger.debug(f"{nimages} images were read.")

            # Select from the original images set only those in modified_images
            original_images = original_images[index_test]

            # Because the reading from the directory does not garantee the order
            # of the index of the images, it is necessary to order the images so
            # the i-th original image is compared to the i-th modified image.
            order = torch.tensor(np.argsort(index_test), dtype=torch.long)
            modified_images = modified_images[order]
            original_images = original_images[order]
            row = [model_name, niter, eps, attacker_name]

            for metric in metrics:
                new_batch_size = 20 if LPIPS in metric[0] else batch_size
                measures = []
                for i in tqdm(
                    range(0, nimages, new_batch_size),
                    desc=f"Obtaining {metric[0]}",
                    unit="batches",
                    leave=False,
                ):
                    mod_batch = modified_images[i : i + new_batch_size].to(device)
                    orig_batch = original_images[i : i + new_batch_size].to(device)
                    out = metric[1](mod_batch, orig_batch)

                    if out.dim() == 0:
                        out = out.unsqueeze(0)

                    measures.append(out.cpu())

                    if device == "cuda":
                        del mod_batch, orig_batch
                        torch.cuda.empty_cache()

                measures = torch.cat(measures)
                row.extend([measures.mean(dim=0).item(), measures.std(dim=0).item()])
            row.append(testimagesdir)
            rows.append(row)

    with open(args.output_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
