# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="name of dataset")

    parser.add_argument(
        "--dataset", type=str, action="append", default=None, help="name of dataset"
    )
    args = parser.parse_args()

    seeds = [111, 333, 555]

    blender_scene_list = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]

    llff_scene_list = [
        "fern",
        "flower",
        "fortress",
        "horns",
        "orchids",
        "leaves",
        "room",
        "trex",
    ]

    tnt_scene_list = [
        "tat_intermediate_M60",
        "tat_intermediate_Playground",
        "tat_intermediate_Train",
        "tat_training_Truck",
    ]

    lf_scene_list = [
        "africa",
        "basket",
        "ship",
        "statue",
        "torch",
    ]

    file_name = f"{args.model}.sh"
    file = open(os.path.join("../scripts", file_name), "w")

    scene_list = []

    if "blender" in args.dataset:
        ginc = f"configs/{args.model}/blender.gin"
        for scene in blender_scene_list:
            for seed in seeds:
                file.write(
                    f"python3 -m run --ginc {ginc} --scene_name {scene} --seed {seed}\n"
                )

    if "llff" in args.dataset:
        ginc = f"configs/{args.model}/llff.gin"
        for scene in llff_scene_list:
            for seed in seeds:
                file.write(
                    f"python3 -m run --ginc {ginc} --scene_name {scene} --seed {seed}\n"
                )

    if "tanks_and_temples" in args.dataset or "tnt" in args.dataset:
        ginc = f"configs/{args.model}/tnt.gin"
        for scene in tnt_scene_list:
            for seed in seeds:
                file.write(
                    f"python3 -m run --ginc {ginc} --scene_name {scene} --seed {seed}\n"
                )

    if "lf" in args.dataset:
        ginc = f"configs/{args.model}/lf.gin"
        for scene in lf_scene_list:
            for seed in seeds:
                file.write(
                    f"python3 -m run --ginc {ginc} --scene_name {scene} --seed {seed}\n"
                )

    file.close()
