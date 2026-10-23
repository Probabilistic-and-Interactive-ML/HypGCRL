# Adapted from OGBench (https://github.com/seohongpark/ogbench),
# Copyright (c) 2024 OGBench Authors, MIT License.
# Modified for this repository in 2026.
"""Environment and dataset helpers adapted from OGBench (MIT license)."""

import os

import ogbench

from jaxgcrl.offline.datasets import Dataset


def make_env_and_datasets(dataset_name, dataset_dir="~/.ogbench/data"):
    """Make OGBench environment and datasets."""
    # Use compact dataset to save memory.
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, dataset_dir=os.path.expanduser(dataset_dir), compact_dataset=True
    )
    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset)
    env.reset()
    return env, train_dataset, val_dataset
