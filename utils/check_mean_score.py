# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import json
import os

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--postfix", default="", type=str, help="files name to filter")
    parser.add_argument(
        "--dirpath", default=".", type=str, help="path to the directory"
    )
    args = parser.parse_args()

    json_list = []
    for dirname in os.listdir(args.dirpath):
        json_path = os.path.join(args.dirpath, dirname, "results.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as fp:
                json_list.append(json.load(fp))

    print(len(json_list))
    psnr_mean, psnr_train, psnr_val, psnr_test = [], [], [], []
    ssim_mean, ssim_train, ssim_val, ssim_test = [], [], [], []
    lpips_mean, lpips_train, lpips_val, lpips_test = [], [], [], []

    for json_file in json_list:
        psnr_mean.append(json_file["PSNR"]["mean"])
        psnr_train.append(json_file["PSNR"]["train_mean"])
        psnr_val.append(json_file["PSNR"]["val_mean"])
        psnr_test.append(json_file["PSNR"]["test_mean"])
        ssim_mean.append(json_file["SSIM"]["mean"])
        ssim_train.append(json_file["SSIM"]["train_mean"])
        ssim_val.append(json_file["SSIM"]["val_mean"])
        ssim_test.append(json_file["SSIM"]["test_mean"])
        lpips_mean.append(json_file["LPIPS-VGG"]["mean"])
        lpips_train.append(json_file["LPIPS-VGG"]["train_mean"])
        lpips_val.append(json_file["LPIPS-VGG"]["val_mean"])
        lpips_test.append(json_file["LPIPS-VGG"]["test_mean"])

    score_name = (
        "psnr_mean",
        "psnr_train",
        "psnr_val",
        "psnr_test",
        "ssim_mean",
        "ssim_train",
        "ssim_val",
        "ssim_test",
        "lpips_mean",
        "lpips_train",
        "lpips_val",
        "lpips_test",
    )

    for name in score_name:
        print(f"{name} : {np.array(eval(name)).mean()}")
