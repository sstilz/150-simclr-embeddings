import math
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import get_data
from src.device import get_device
from src.models import ConvNet


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--use_gpu", type=bool, default=True)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")

    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--proj_head_width", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=1_024)

    # GitHub has a max file size of 100MB, which translates to 2.5e7 32-bit floats.
    parser.add_argument("--max_array_size", type=int, default=int(2e7))

    return parser.parse_args()


@torch.inference_mode()
def encode(loader: DataLoader, encoder: ConvNet, device: str) -> np.ndarray:
    embeddings = []

    for images_i, _ in loader:
        images_i = images_i.to(device)
        embeddings_i = encoder(images_i)
        embeddings += [embeddings_i.cpu().numpy()]

    return np.concatenate(embeddings)


def main(cfg: Namespace) -> None:
    device = get_device(cfg.use_gpu)

    data_dir = Path(cfg.data_dir) / cfg.dataset
    model_dir = Path(cfg.model_dir) / cfg.dataset

    dataset_class, input_shape, _, test_transform = get_data(cfg.dataset, cfg.data_dir)

    state_dict = torch.load(model_dir / "encoder.pth")

    encoder = ConvNet(input_shape=input_shape, output_size=cfg.embedding_size)
    encoder = encoder.to(device)
    encoder.load_state_dict(state_dict)

    for subset in ("train", "test"):
        dataset = dataset_class(train=(subset == "train"), transform=test_transform)
        loader = DataLoader(dataset, batch_size=cfg.batch_size)

        embeddings = encode(loader, encoder, device)

        if embeddings.size > cfg.max_array_size:
            n_splits = math.ceil(embeddings.size / cfg.max_array_size)

            for i, embeddings_i in enumerate(np.array_split(embeddings, n_splits, axis=0)):
                filepath = data_dir / f"embeddings_simclr_{subset}_part{i + 1}of{n_splits}.npy"
                np.save(filepath, embeddings_i, allow_pickle=False)

        else:
            np.save(data_dir / f"embeddings_simclr_{subset}.npy", embeddings, allow_pickle=False)

        np.save(data_dir / f"labels_{subset}.npy", dataset.targets, allow_pickle=False)


if __name__ == "__main__":
    cfg = get_config()
    main(cfg)
