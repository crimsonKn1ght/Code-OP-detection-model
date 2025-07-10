import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(image, radii=[1, 2, 3], P=8):
    if image.ndim == 3:
        image = image.mean(axis=0)
    img_uint8 = ((image - image.min()) * (255.0 / (image.max() - image.min() + 1e-8))).astype(np.uint8)

    histograms = []
    for R in radii:
        lbp_ri = local_binary_pattern(img_uint8, P, R, method="uniform")
        hist_ri, _ = np.histogram(lbp_ri, bins=P+2, range=(0, P+2))
        histograms.append(hist_ri / (hist_ri.sum() + 1e-8))

        var_map = np.zeros_like(img_uint8, dtype=float)
        for i in range(1, img_uint8.shape[0]-1):
            for j in range(1, img_uint8.shape[1]-1):
                window = img_uint8[i-1:i+2, j-1:j+2]
                var_map[i,j] = np.var(window)
        var_hist, _ = np.histogram(var_map, bins=10, range=(0, var_map.max()+1e-8))
        histograms.append(var_hist / (var_hist.sum() + 1e-8))

    return np.concatenate(histograms)