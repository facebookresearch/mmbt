#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

import torch
import torch.nn as nn


class GloveBowEncoder(nn.Module):
    def __init__(self, args):
        super(GloveBowEncoder, self).__init__()
        self.args = args
        self.embed = nn.Embedding(args.vocab_sz, args.embed_sz)
        self.load_glove()
        self.embed.weight.requires_grad = False

    def load_glove(self):
        print("Loading glove")
        pretrained_embeds = np.zeros(
            (self.args.vocab_sz, self.args.embed_sz), dtype=np.float32
        )
        for line in open(self.args.glove_path):
            w, v = line.split(" ", 1)
            if w in self.args.vocab.stoi:
                pretrained_embeds[self.args.vocab.stoi[w]] = np.array(
                    [float(x) for x in v.split()], dtype=np.float32
                )
        self.embed.weight = torch.nn.Parameter(torch.from_numpy(pretrained_embeds))

    def forward(self, x):
        return self.embed(x).sum(1)


class GloveBowClf(nn.Module):
    def __init__(self, args):
        super(GloveBowClf, self).__init__()
        self.args = args
        self.enc = GloveBowEncoder(args)
        self.clf = nn.Linear(args.embed_sz, args.n_classes)

    def forward(self, x):
        return self.clf(self.enc(x))
