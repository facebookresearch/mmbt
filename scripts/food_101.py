#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
import re
import shutil
import sys

from os.path import join
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm


def format_food101_dataset(dataset_root_path):
    print("Parsing data...")
    data = parse_data(dataset_root_path)
    data["train"], data["dev"] = train_test_split(
        data["train"],
        test_size=5000,
        stratify=[x["label"] for x in data["train"]],
    )
    print("Saving everything into format...")
    save_in_format(data, dataset_root_path)


def format_txt_file(content):
    for c in '<>/\\+=-_[]{}\'\";:.,()*&^%$#@!~`':
        content = content.replace(c, ' ')
    content = re.sub("\s\s+" , ' ', content)
    return content.lower().replace("\n", " ")


def parse_data(source_dir):
    splits = ["train", "test"]
    data = {split: [] for split in splits}
    for split in splits:
        for label in os.listdir(join(source_dir, "images", split)):
            for img in os.listdir(join(source_dir, "images", split, label)):
                match = re.search(
                    r"(?P<name>\w+)_(?P<num>[\d-]+)\.(?P<ext>\w+)", img
                )
                num = match.group("num")
                dobj = {}
                dobj["id"] = label + "_" + img
                dobj["label"] = label
                txt_path = join(
                    source_dir, "texts_txt", label, "{}_{}.txt".format(label, num)
                )
                if not os.path.exists(txt_path):
                    continue
                dobj["text"] = format_txt_file(open(txt_path).read())
                dobj["img"] = join("images", split, label, img)
                data[split].append(dobj)
    return data


def save_in_format(data, target_path):
    """
    Stores the data to @target_dir. It does not store metadata.
    """

    for split_name in data:
        jsonl_loc = join(target_path, split_name + ".jsonl")
        with open(jsonl_loc, "w") as jsonl:
            for sample in tqdm(data[split_name]):
                jsonl.write("%s\n" % json.dumps(sample))


if __name__ == "__main__":
    # Path to the directory for Food101
    format_food101_dataset(sys.argv[1])

