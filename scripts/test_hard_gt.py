#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Script for making "test_hard_gt.jsonl"
## https://github.com/facebookresearch/mmbt/issues/6

import argparse
import copy
import json
import os

def get_args(parser):
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--mmdb_path", type=str, default="/home/ubuntu/data/mmimdb/mmimdb")
    parser.add_argument("--food_path", type=str, default="/home/ubuntu/script/food101")
    parser.add_argument("--hard_gt_ids_path", type=str, default="hard_gt_ids.json")


def get_hard_gt(args):
    # load id list
    with open(args.hard_gt_ids_path) as f:
        hard_gt_ids = json.load(f)
    # get id list for specified task
    for hard_gt_dict in hard_gt_ids:
        if hard_gt_dict["task"]==args.task:
            hard_gt_id = hard_gt_dict["hard_gt"]
            break
        else:
            pass
    # load test.jsonl
    if args.task == "mmimdb":
        path4task = copy.deepcopy(args.mmdb_path)
    elif args.task == "food101":
        path4task = copy.deepcopy(args.mmdb_path)
    else:
        pass
    with open(os.path.join(path4task, "test.jsonl")) as f:
        tests = [json.loads(line) for line in f]
    # append lines of test.jsonl containing id in specified list to test_hard_gt.jsonl
    with open(os.path.join(path4task, "test_hard_gt.jsonl"), "w") as f:
        for test in tests:
            if test["id"] in hard_gt_id:
                json.dump(test, f)
                f.write('\n')
            else:
                pass
#     test_hard_gt = []
#     for test in tests:
#         if test["id"] in hard_gt_id:
#             test_hard_gt.append(test)
#         else:
#             pass
#     return test_hard_gt

def cli_main():
    parser = argparse.ArgumentParser(description="Create Hard GT json")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    get_hard_gt(args)
    #return hard_gt


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
