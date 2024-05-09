from functools import partial
from pathlib import Path
from subprocess import check_call
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from lightly.transforms import SimCLRTransform
from numpy.lib.npyio import NpzFile
from numpy.random import Generator
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, VisionDataset
from torchvision.transforms import Compose, Normalize, ToTensor


def get_next(dataloader: DataLoader) -> Union[Tensor, Tuple]:
    try:
        return next(dataloader)
    except:
        dataloader = iter(dataloader)
        return next(dataloader)


def get_data(dataset: str, data_dir: Union[str, Path]) -> Tuple[partial, list, Compose, Compose]:
    if dataset == "dsprites":
        dataset = partial(DSprites, data_dir=(Path(data_dir) / "dsprites"))

        input_shape = [64, 64]

        train_transform = SimCLRTransform(
            input_size=64,
            min_scale=0.75,
            hf_prob=0.5,
            rr_prob=0.5,
            vf_prob=0.5,
            normalize=dict(mean=0.0425, std=0.2017),
        )

        test_transform = Compose([ToTensor(), Normalize(mean=0.0425, std=0.2017)])

    elif dataset == "mnist":
        dataset = partial(MNIST, root=(Path(data_dir) / "mnist"), download=True)

        input_shape = [28, 28]

        train_transform = SimCLRTransform(
            input_size=28,
            min_scale=0.5,
            hf_prob=0,
            rr_prob=0,
            vf_prob=0,
            normalize=dict(mean=0.1307, std=0.3081),
        )

        test_transform = Compose([ToTensor(), Normalize(mean=0.1307, std=0.3081)])

    else:
        raise NotImplementedError

    return dataset, input_shape, train_transform, test_transform


def split_indices_using_class_labels(
    labels: np.ndarray,
    test_size: Union[dict, float, int],
    rng: Generator,
    balance_test_set: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    sklearn.model_selection.train_test_split() doesn't produce exact stratification.
    """
    if isinstance(test_size, float) and balance_test_set:
        class_counts = np.bincount(labels)
        test_size = int(test_size * min(class_counts))
        test_size = max(test_size, 1)

    train_inds, test_inds = [], []

    for _class in np.unique(labels):
        class_inds = np.flatnonzero(labels == _class)
        class_inds = rng.permutation(class_inds)

        if isinstance(test_size, dict):
            class_test_size = test_size[_class]
        else:
            class_test_size = test_size

        if isinstance(class_test_size, float):
            class_test_size = int(class_test_size * len(class_inds))
            class_test_size = max(class_test_size, 1)

        train_inds += [class_inds[:-class_test_size]]
        test_inds += [class_inds[-class_test_size:]]

    train_inds = np.concatenate(train_inds)
    train_inds = rng.permutation(train_inds)

    test_inds = np.concatenate(test_inds)
    test_inds = rng.permutation(test_inds)

    return train_inds, test_inds


class DSprites(VisionDataset):
    """
    DSprites dataset with one of the six underlying latents as the class label. The dataset is split
    into training and test sets. The split depends on three parameters: the latent used as the class
    label, the fraction of the dataset used for testing, and the random seed.

    References:
        https://github.com/google-deepmind/dsprites-dataset
    """

    filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    target_names = {"color": 0, "shape": 1, "scale": 2, "orientation": 3, "pos_x": 4, "pos_y": 5}

    def __init__(
        self,
        data_dir: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_fraction: float = 0.15,
        target_latent: str = "shape",
        seed: int = 0,
    ) -> None:
        super().__init__(data_dir, transform=transform, target_transform=target_transform)

        data_dir = Path(data_dir)

        if not (data_dir / self.filename).exists():
            self.download(data_dir)

        npz_file = np.load(data_dir / self.filename)
        inputs, labels = self.process_npz(npz_file, target_latent)

        rng = np.random.default_rng(seed=seed)
        train_inds, test_inds = split_indices_using_class_labels(labels, test_fraction, rng)

        if train:
            self.data = inputs[train_inds]
            self.targets = labels[train_inds]
        else:
            self.data = inputs[test_inds]
            self.targets = labels[test_inds]

        self.data = torch.tensor(self.data)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        References:
            https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
            https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
        """
        image, target = self.data[index], int(self.targets[index])

        image = Image.fromarray(image.numpy(), mode="L")  # Each pixel is an 8-bit integer

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.data)

    def download(self, data_dir: Path) -> None:
        url = f"https://github.com/google-deepmind/dsprites-dataset/raw/master/{self.filename}"
        data_dir.mkdir(parents=True, exist_ok=True)
        check_call(["curl", "--location", "--output", data_dir / self.filename, url])

    def process_npz(self, npz_file: NpzFile, target_latent: str) -> Tuple[np.ndarray, np.ndarray]:
        inputs = npz_file["imgs"]  # [N, 64, 64], np.uint8, values in {0, 1}
        inputs *= 255  # [N, 64, 64], np.uint8, values in {0, 255}

        labels = npz_file["latents_classes"]  # [N, 6], np.int64
        labels = labels[:, self.target_names[target_latent]]  # [N,]

        return inputs, labels  # [N, 64, 64], [N,]
