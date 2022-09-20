import os

import imageio
import numpy as np
from PIL import Image

if __name__ == "__main__":
    img_dir = "./data/refnerf/ball"

    for i in range(100):
        alpha = imageio.imread(os.path.join(img_dir, "train", f"r_{i}_alpha.png"))
        img = imageio.imread(os.path.join(img_dir, "train", f"r_{i}.png"))
        img = np.concatenate([img, alpha[..., None]], axis=-1)
        img = Image.fromarray(img)
        img.save(os.path.join(img_dir, "train", f"r_{i}.png"))

    for i in range(200):
        alpha = imageio.imread(os.path.join(img_dir, "test", f"r_{i}_alpha.png"))
        img = imageio.imread(os.path.join(img_dir, "test", f"r_{i}.png"))
        img = np.concatenate([img, alpha[..., None]], axis=-1)
        img = Image.fromarray(img)
        img.save(os.path.join(img_dir, "test", f"r_{i}.png"))
