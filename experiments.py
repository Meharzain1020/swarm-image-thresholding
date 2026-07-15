import os
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import data, color

from src.fitness_functions import compute_histogram, otsu_fitness, kapur_fitness
from src.optimizers import ParticleSwarmOptimizer, GeneticAlgorithm
from src.image_utils import preprocess_image, apply_multilevel_threshold
from src.metrics import compute_psnr, compute_ssim

def find_plateau(history, tolerance=1e-4):
    """Finds iteration where fitness plateaus."""
    max_val = history[-1]
    for idx, val in enumerate(history):
        if abs(max_val - val) / (abs(max_val) + 1e-8) <= tolerance:
            return idx + 1
    return len(history)

def run_experiments():
    os.makedirs('outputs/experiment_result', exist_ok=True)
    
    benchmark_images = {
        'Camera': data.camera(),
        'Coins': data.coins(),
        'Astronaut': color.rgb2gray(data.astronaut()) * 255,
        'Moon': data.moon(),
        'Checkerboard': (data.checkerboard() * 255).astype(np.uint8)
    }

    results = []
    num_thresholds = 4
    max_iter = 50

    for name, raw_img in benchmark_images.items():
        image = preprocess_image(raw_img.astype(np.uint8))
        probabilities = compute_histogram(image)

        t0 = time.time()
        pso = ParticleSwarmOptimizer(30, num_thresholds, max_iter, probabilities, otsu_fitness)
        t_pso, hist_pso = pso.optimize()
        time_pso = time.time() - t0
        seg_pso = apply_multilevel_threshold(image, t_pso)
        plat_pso = find_plateau(hist_pso)

        t0 = time.time()
        ga = GeneticAlgorithm(30, num_thresholds, max_iter, probabilities, otsu_fitness)
        t_ga, hist_ga = ga.optimize()
        time_ga = time.time() - t0
        seg_ga = apply_multilevel_threshold(image, t_ga)
        plat_ga = find_plateau(hist_ga)

        results.append({
            'Image': name,
            'PSO_PSNR': compute_psnr(image, seg_pso),
            'PSO_SSIM': compute_ssim(image, seg_pso),
            'PSO_Time(s)': time_pso,
            'PSO_Plateau_Iter': plat_pso,
            'GA_PSNR': compute_psnr(image, seg_ga),
            'GA_SSIM': compute_ssim(image, seg_ga),
            'GA_Time(s)': time_ga,
            'GA_Plateau_Iter': plat_ga,
        })

        plt.figure(figsize=(8, 4))
        plt.plot(hist_pso, label='PSO Otsu', color='blue')
        plt.plot(hist_ga, label='GA Otsu', color='orange')
        plt.axvline(plat_pso, color='blue', linestyle='--', alpha=0.7, label=f'PSO Plateau ({plat_pso})')
        plt.axvline(plat_ga, color='orange', linestyle='--', alpha=0.7, label=f'GA Plateau ({plat_ga})')
        plt.title(f'Convergence Comparison - {name}')
        plt.xlabel('Iteration')
        plt.ylabel('Otsu Fitness Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'outputs/experiment_result/{name.lower()}_convergence.png')
        plt.close()

    df = pd.DataFrame(results)
    df.to_csv('outputs/experiment_result/results_summary.csv', index=False)
    
    print("\n================ BENCHMARK SUMMARY TABLE ================")
    print(df.to_string(index=False))
    print("=========================================================\n")

if __name__ == '__main__':
    run_experiments()