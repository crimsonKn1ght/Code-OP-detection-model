import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm(
        image, 
        distances=[1, 2, 3], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'entropy']
        ):

        if image.ndim == 3:
            image = image.mean(axis=0)
        img_uint8 = ((image - image.min()) * (255.0 / (image.max() - image.min() + 1e-8))).astype(np.uint8)

        glcm = graycomatrix(img_uint8, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        feature_vector = []

        for prop in properties:
            if prop == 'entropy':
                glcm_prob = glcm / (glcm.sum() + 1e-8)
                entropy = -np.sum(glcm_prob * np.log(glcm_prob + 1e-8))
                feature_vector.append(entropy)
            else:
                vals = graycoprops(glcm, prop)
                feature_vector.extend(vals.ravel())

        return np.array(feature_vector)