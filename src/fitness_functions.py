import numpy as np

def compute_histogram(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    return hist / hist.sum()

def otsu_fitness(thresholds: np.ndarray, probabilities: np.ndarray) -> float:
    
    thresholds = np.sort(np.round(thresholds).astype(int))
    bins = np.concatenate(([0], thresholds, [256]))
    
    total_mean = np.sum(np.arange(256) * probabilities)
    variance = 0.0

    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i+1]
        weight = np.sum(probabilities[start:end])
        if weight == 0:
            continue
        mean = np.sum(np.arange(start, end) * probabilities[start:end]) / weight
        variance += weight * ((mean - total_mean) ** 2)

    return variance

def kapur_fitness(thresholds: np.ndarray, probabilities: np.ndarray) -> float:
    
    thresholds = np.sort(np.round(thresholds).astype(int))
    bins = np.concatenate(([0], thresholds, [256]))
    
    total_entropy = 0.0
    epsilon = 1e-10

    for i in range(len(bins) - 1):
        start, end = bins[i], bins[i+1]
        weight = np.sum(probabilities[start:end])
        if weight <= epsilon:
            continue
        segment_prob = probabilities[start:end] / weight
        entropy = -np.sum(segment_prob * np.log(segment_prob + epsilon))
        total_entropy += entropy

    return total_entropy