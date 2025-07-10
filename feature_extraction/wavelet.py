import pywt
import numpy as np
from scipy.stats import skew, kurtosis

def extract_wavelet(image, wavelet='db1', level=2):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    features = []
    for coeff in coeffs[1:]:
        for arr in coeff:
            features.extend([
                arr.mean(),
                arr.std(),
                skew(arr.ravel()),
                kurtosis(arr.ravel())
            ])
    return np.array(features)