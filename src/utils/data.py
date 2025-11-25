"""
Data related functionalities
"""

import math
import os
from typing import Callable, Optional, Union, cast
from pathlib import Path

import timm
import torch
from torch import nn
import torchvision.transforms.v2 as transforms
from torchvision.datasets import ImageNet
from torchvision.datasets.folder import has_file_allowed_extension

THREAT_MODEL = "Linf"
DATASET_NAME = "imagenet"

# Applied trasnformations to the subset of ImageNet and the modified images
INITIAL_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float, True),    
])

class LimitedImageNet(ImageNet):
    """
    Subset of ImageNet that is selected by specifying the name of the
    desried images in a text file.
    """

    @property
    def nb_classes(self):
        """Get the number of classes in the dataset."""
        return 1000

    @staticmethod
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: dict[str, int],
        extensions: Optional[tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> list[tuple[str, int]]:
        """
        Generates a list of samples of a form (path_to_sample, class) that
        are specified in the file 'helper_files/imagenet_test_image_ids.txt'.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """

        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)
        instances = []

        # TODO: no hard-code
        with open('src/helper_files/imagenet_test_image_ids.txt', 'r') as f:
            
            for fname in f.readlines():
                fname = fname.strip()
                path = os.path.join(directory, fname)
                if is_valid_file(path):
                    class_index = class_to_idx[fname.split('/')[0]]
                    item = path, class_index
                    instances.append(item)

        return instances


def get_preprocessing(model_name):
    timm_model_name = f"{model_name}_{DATASET_NAME}_{THREAT_MODEL}"
    if not timm.is_model(timm_model_name):
        raise ValueError(f"{timm_model_name} is not available in the package timm.")

    model = timm.create_model(model_name)
    if isinstance(model, nn.Sequential):
        # Normalization has been applied, take the inner model to get the other info
        model = model.model
    interpolation = model.default_cfg["interpolation"]
    crop_pct = model.default_cfg["crop_pct"]
    img_size = model.default_cfg["input_size"][1]
    scale_size = int(math.floor(img_size / crop_pct))
    return transforms.Compose(
        [
            INITIAL_TRANSFORM,
            transforms.Resize(
                scale_size, interpolation=transforms.InterpolationMode(interpolation)
            ),
            transforms.CenterCrop(img_size), 
            transforms.Lambda(lambda x: x.clamp(0.0, 1.0)) # porque lpips da fallo al comparar tensor(1.00000) y 1.0
        ]
    )
