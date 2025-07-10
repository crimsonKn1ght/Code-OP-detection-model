import numpy as np


def extract_lacunarity(image, box_sizes=[5, 10, 20]):
    if image.ndim == 3:
        image = image.mean(axis=0)
    if image.size == 0:
        return np.zeros(len(box_sizes))
    return np.array([np.mean(image) * size for size in box_sizes])
