"""
Utility functions for Epitome experiments.
"""
from typing import List, Mapping, Any, Tuple

import os
import json
import random
import copy
from nltk.tokenize import sent_tokenize
from src.featurizers.BertFeaturizer import BertFeaturizer
from empathy.constants import EMP_CONFIGS, EMP_TASKS as TASKS

MM_HOME = os.environ.get("MM_HOME")


def setup_emp_config(mm_data):
    """set up orchestrator for featurization"""
    mm_configs = copy.deepcopy(EMP_CONFIGS)
    for instance in mm_data:
        for idx, task in enumerate(TASKS):
            config = mm_configs[idx]
            assert config["name"] == "empathy_%s" % task
            level = instance[task]["level"]
            if level != "0":
                rationales = instance[task]["rationales"].split("|")
                rationales = [x for x in rationales if x != ""]
                config["setup_args"]["seed"].extend(rationales)
    return mm_configs


def init_featurizer(config) -> BertFeaturizer:
    """Set up orchestrator for featurization"""
    orchestrator = BertFeaturizer(MM_HOME, config)
    return orchestrator


def load_emp_data(
    data_path, seed, **kwargs
) -> Tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """
    Load Epitome data, including sentence-tokenizing the input text and
    splitting the data into train, val, and test splits.
    """
    train_ratio = kwargs.get("train_ratio", 0.7)
    val_ratio = kwargs.get("val_ratio", 0.15)
    train_seed_ratio = kwargs.get("train_seed_ratio", 0)

    with open(data_path, "r") as file_p:
        data = json.load(file_p)

    for _, data_obj in data.items():
        data_obj["response_tokenized"] = list(
            sent_tokenize(data_obj["response_post"])
        )

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    train_seed_size = int(train_size * train_seed_ratio)

    keys = list(data.keys())
    random.seed(seed)
    random.shuffle(keys)

    train_keys = keys[:train_size]
    train_seed_keys = keys[:train_seed_size]
    train_keys = [key for key in train_keys if key not in train_seed_keys]
    val_keys = keys[train_size : train_size + val_size]
    test_keys = keys[train_size + val_size :]

    train = [data[key] for key in train_keys]
    train_seed = [data[key] for key in train_seed_keys]
    val = [data[key] for key in val_keys]
    test = [data[key] for key in test_keys]

    return train, val, test, train_seed


def save_json(data, filepath):
    """ Save data to filepath """
    with open(filepath, "w") as file_p:
        json.dump(data, file_p, indent=2)


def load_json(filepath):
    """ Load data from filepath """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)
    return data
