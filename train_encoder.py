import logging
from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from torch.optim import SGD
from torch.utils.data import DataLoader

from src.data import get_data, get_next
from src.device import get_device
from src.logging import save_table, set_up_logging
from src.models import ConvNet
from src.random import get_rng


def get_config() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")

    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--proj_head_width", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=1_024)
    parser.add_argument("--learning_rate", type=float, default=1)
    parser.add_argument("--log_gap", type=int, default=100)
    parser.add_argument("--n_optim_steps", type=int, default=50_000)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    return parser.parse_args()


def main(cfg: Namespace) -> None:
    """
    References:
        https://docs.lightly.ai/getting_started/lightly_at_a_glance.html
        https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html
    """
    # ----------------------------------------------------------------------------------------------
    logging.info("Setting up")
    logging.info(f"Seed: {cfg.seed}")

    _ = get_rng(cfg.seed)
    device = get_device(cfg.use_gpu)

    if cfg.use_gpu and (device not in {"cuda", "mps"}):
        logging.warning(f"Device: {device}")
    else:
        logging.info(f"Device: {device}")

    model_dir = Path(cfg.model_dir) / cfg.dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(cfg.results_dir) / cfg.dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_class, input_shape, train_transform, _ = get_data(cfg.dataset, cfg.data_dir)

    dataset = dataset_class(train=True)
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=train_transform)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    encoder = ConvNet(input_shape=input_shape, output_size=cfg.embedding_size)
    encoder = encoder.to(device)

    proj_head = SimCLRProjectionHead(input_dim=cfg.embedding_size, hidden_dim=cfg.proj_head_width)
    proj_head = proj_head.to(device)

    loss_fn = NTXentLoss()

    params = chain(encoder.parameters(), proj_head.parameters())
    optimizer = SGD(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # ----------------------------------------------------------------------------------------------
    logging.info("Training")

    steps, losses = [], []

    for step in range(cfg.n_optim_steps):
        (inputs_0, inputs_1), _, _ = get_next(loader)

        embeddings_0 = proj_head(encoder(inputs_0.to(device)))
        embeddings_1 = proj_head(encoder(inputs_1.to(device)))

        loss = loss_fn(embeddings_0, embeddings_1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps += [step]
        losses += [loss.item()]

        if (step > 0) and (step % cfg.log_gap == 0):
            mean_loss = np.mean(losses[-cfg.log_gap :])
            logging.info(f"Step {step:05}: loss = {mean_loss:.4f}")

    # ----------------------------------------------------------------------------------------------
    logging.info("Saving results")

    train_log = pd.DataFrame({"step": steps, "loss": losses})
    formatting = {"step": "{:05}".format, "loss": "{:.4f}".format}
    save_table(results_dir / "pretraining.csv", train_log, formatting)

    torch.save(encoder.state_dict(), model_dir / "encoder.pth")
    torch.save(proj_head.state_dict(), model_dir / "projection_head.pth")


if __name__ == "__main__":
    set_up_logging()
    cfg = get_config()
    main(cfg)
