import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(image, (3, 3), 0)

def apply_multilevel_threshold(image: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    bins = np.concatenate(([0], thresholds, [256]))
    segmented = np.zeros_like(image)
    for i in range(len(bins) - 1):
        mask = (image >= bins[i]) & (image < bins[i+1])
        if np.any(mask):
            segmented[mask] = int(np.mean(image[mask]))
    return segmented