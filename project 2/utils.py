#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2021 mei 10, 23:35:15
@last modified : 2021 mei 10, 23:54:28
"""


def get_latest_ckpt(path: str):
    import re
    import os
    from glob import glob

    all_ckpt_fnames = [
        fname
        for fname in os.listdir(path)
        if re.match("^checkpoint_[0-9]+$", fname)
        and os.path.isdir(os.path.join(path, fname))
    ]

    all_ckpt_fnames.sort(key=lambda fname: int(fname.split("_")[-1]))

    latest_ckpt = all_ckpt_fnames[-1]

    ckpt_path = os.path.join(path, latest_ckpt)

    all_subpath = os.listdir(ckpt_path)

    for latest_ckpt_path in all_subpath:
        if re.match("^checkpoint-[0-9]+$", latest_ckpt_path):
            return os.path.join(ckpt_path, latest_ckpt_path)
