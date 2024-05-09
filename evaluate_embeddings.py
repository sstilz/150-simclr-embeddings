from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.logging import save_table
from src.random import get_rng


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--n_labels_end", type=int, default=500)

    parser.add_argument("--inverse_l2_coef", type=float, default=1)
    parser.add_argument("--n_optim_steps_max", type=int, default=1_000)

    return parser.parse_args()


def load_embeddings(data_dir: Path, subset: str) -> np.ndarray:
    embeddings_paths = sorted(data_dir.glob(f"embeddings_simclr_{subset}*.npy"))
    return np.concatenate([np.load(path) for path in embeddings_paths])


def sample_n_per_class(labels: np.ndarray, n_per_class: int, rng: Generator) -> np.ndarray:
    sampled_inds = []

    for _class in np.unique(labels):
        class_inds = np.flatnonzero(labels == _class)
        sampled_inds += [rng.choice(class_inds, size=n_per_class, replace=False)]

    sampled_inds = np.concatenate(sampled_inds)
    sampled_inds = rng.permutation(sampled_inds)

    return sampled_inds


def compute_mean_and_std(
    x: np.ndarray, axis: int = 0, keepdims: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    To avoid dividing by zero we set the standard deviation to one if it is less than epsilon.
    """
    mean = np.mean(x, axis=axis, keepdims=keepdims)
    std = np.std(x, axis=axis, keepdims=keepdims)

    eps = np.finfo(x.dtype).eps

    if isinstance(std, np.ndarray):
        std[std < eps] = 1
    elif std < eps:
        std = 1

    return mean, std


def main(cfg: Namespace) -> None:
    rng = get_rng(seed=cfg.seed)

    data_dir = Path(cfg.data_dir) / cfg.dataset
    results_dir = Path(cfg.results_dir) / cfg.dataset

    train_inputs = load_embeddings(data_dir, subset="train")
    train_labels = np.load(data_dir / "labels_train.npy")

    test_inputs = load_embeddings(data_dir, subset="test")
    test_labels = np.load(data_dir / "labels_test.npy")

    breakpoint()

    mean, std = compute_mean_and_std(train_inputs)
    train_inputs = (train_inputs - mean) / std
    test_inputs = (test_inputs - mean) / std

    n_per_class = 1
    n_labels, accs, logliks = [], [], []

    while True:
        train_inds = sample_n_per_class(train_labels, n_per_class, rng)

        model = LogisticRegression(C=cfg.inverse_l2_coef, max_iter=cfg.n_optim_steps_max)
        model.fit(train_inputs[train_inds], train_labels[train_inds])

        probs = model.predict_proba(test_inputs)
        acc = model.score(test_inputs, test_labels)
        loglik = -log_loss(test_labels, probs)

        n_labels += [len(train_inds)]
        accs += [acc]
        logliks += [loglik]

        if len(train_inds) >= cfg.n_labels_end:
            break
        else:
            n_per_class += 1

    test_log = pd.DataFrame({"n_labels": n_labels, "test_acc": accs, "test_loglik": logliks})
    formatting = {
        "n_labels": "{:03}".format,
        "test_acc": "{:.4f}".format,
        "test_loglik": "{:.4f}".format,
    }
    save_table(results_dir / f"testing_seed{cfg.seed:02}.csv", test_log, formatting)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
