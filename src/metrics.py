import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_psnr(original: np.ndarray, segmented: np.ndarray) -> float:
    mse = np.mean((original.astype(float) - segmented.astype(float)) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(original: np.ndarray, segmented: np.ndarray) -> float:
    return ssim(original, segmented, data_range=255)