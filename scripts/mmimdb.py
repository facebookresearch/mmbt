#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
import sys


def format_mmimdb_dataset(dataset_root_path):
    train_label_set = set()
    is_save_sample = True
    with open(os.path.join(dataset_root_path, "mmimdb/split.json")) as fin:
        data_splits = json.load(fin)
    for split_name in data_splits:
        with open(os.path.join(dataset_root_path, split_name + ".jsonl"), "w") as fw:
            for idx in data_splits[split_name]:
                with open(os.path.join(dataset_root_path, "mmimdb/dataset/{}.json".format(idx))) as fin:
                    data = json.load(fin)
                plot_id = np.array([len(p) for p in data["plot"]]).argmax()
                dobj = {}
                dobj["id"] = idx
                dobj["text"] = data["plot"][plot_id]
                dobj["image"] = "mmimdb/dataset/{}.jpeg".format(idx)
                dobj["label"] = data["genres"]
                if "News" in dobj["label"]:
                    continue
                if split_name == "train":
                    for label in dobj["label"]:
                        train_label_set.add(label)
                else:
                    for label in dobj["label"]:
                        if label not in train_label_set:
                            is_save_sample = False
                if len(dobj["text"]) > 0 and is_save_sample:
                    fw.write("%s\n" % json.dumps(dobj))
                is_save_sample = True


if __name__ == "__main__":
    # Path to the directory for MMIMDB
    format_mmimdb_dataset(sys.argv[1])