#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script for making "test_hard_gt.jsonl"
# https://github.com/facebookresearch/mmbt/issues/6

import argparse
import json
import os


def get_args(parser):
    parser.add_argument(
        "--task", type=str, default="mmimdb", choices=["mmimdb", "food101"]
    )
    parser.add_argument("--path", type=str, default="mmimdb/mmimdb")
    parser.add_argument("--hard_gt_ids", type=str, default="hard_gt_ids.json")


def get_hard_gt(args):
    # load id list
    with open(args.hard_gt_ids) as f:
        hard_gt_ids_dict = json.load(f)
    # get id list for specified task
    hard_gt_ids = hard_gt_ids_dict[args.task]
    # load test.jsonl
    with open(os.path.join(args.path, "test.jsonl")) as f:
        test_jsonl_dicts = [json.loads(line) for line in f]
    # append test.jsonl lines with specified ids to test_hard_gt.jsonl
    with open(os.path.join(args.path, "test_hard_gt.jsonl"), "w") as fw:
        fw.write(
            "\n".join(
                json.dumps(tjd) for tjd in test_jsonl_dicts if tjd["id"] in hard_gt_ids
            )
        )


def cli_main():
    parser = argparse.ArgumentParser(description="Create Hard GT json")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    get_hard_gt(args)


if __name__ == "__main__":
    cli_main()
