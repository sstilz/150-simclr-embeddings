import logging
from pathlib import Path
from typing import Union

from pandas import DataFrame


def set_up_logging() -> None:
    """
    References:
        https://stackoverflow.com/a/44175370
    """
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)s] - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_table(path: Union[Path, str], table: DataFrame, formatting: dict) -> None:
    for key in formatting:
        if key in table:
            table[key] = table[key].apply(formatting[key])

    table.to_csv(path, index=False)
