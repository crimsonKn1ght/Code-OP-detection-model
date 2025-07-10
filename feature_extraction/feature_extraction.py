import numpy as np
from tqdm.auto import tqdm
from feature_extraction.lbp import extract_lbp
from feature_extraction.glcm import extract_glcm
from feature_extraction.wavelet import extract_wavelet
from feature_extraction.lacunarity import extract_lacunarity


def extract_all_features(images):
    features = []
    for img in tqdm(images, desc="Extracting features", leave=False):
        lbp = extract_lbp(img)
        lacunarity = extract_lacunarity(img)
        glcm = extract_glcm(img)
        wavelet = extract_wavelet(img)
        features.append(np.concatenate([lbp, lacunarity, glcm, wavelet]))
    return np.array(features)
